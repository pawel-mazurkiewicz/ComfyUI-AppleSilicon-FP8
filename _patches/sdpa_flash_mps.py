"""Fix: F.scaled_dot_product_attention OOMs / thrashes on MPS at large sequence length.

PyTorch's MPS SDPA materializes the Lq x Lk score matrix, so attention memory
grows O(B*H*Lq*Lk). At 2K-4K token grids (e.g. SeedVR2's DiT upscaling a 4K
frame, or any large DiT) this blows past unified memory and the process is
SIGKILLed — there is no flash-attention / memory-efficient kernel on MPS
(SageAttention / FlashAttention / Triton are all CUDA-only).

Fix: a Metal flash-attention forward kernel (JIT-compiled via
torch.mps.compile_shader), one thread per query row with an online-softmax over
all keys, so the Lq x Lk matrix is NEVER materialized — peak memory drops to
O(B*H*Lq*D). It is gated: only large attentions (per-head area Lq*Lk above a
threshold) route through it; small ones keep stock fused SDPA, which is faster
where it fits. Any unsupported case (additive mask, dropout, D>128, exotic
dtype) or any error falls straight back to the original op — this never crashes.

Supports: MHA + grouped-query (Hkv | Hq), causal (bottom-right aligned),
cross-attn (Lq != Lk), fp16 / bf16 / fp32. fp32 accumulation; inputs upcast.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

TAG = "[AppleSilicon-FP8/sdpa]"

# IMPORTANT: the v0 kernel is MEMORY-bounded but ~3x SLOWER than stock fused SDPA
# (and its fp32 upcast adds copies). So it must ONLY fire when stock SDPA would
# actually OOM — never for attentions that merely happen to be "large" but fit.
# Gate on the would-be score-matrix SIZE (B*Hq*Lq*Lk*2 bytes): only reroute when
# materializing it would genuinely threaten memory. Default 12 GB — well above
# anything a normal diffusion/DiT model produces (so Pixal3D etc. keep the fast
# fused path), catching only pathological global attention. Tunable / disable via
# APPLESILICON_FP8_SDPA_MIN_GB and APPLESILICON_FP8_SDPA=off.
_MIN_SCORE_BYTES = int(float(os.environ.get("APPLESILICON_FP8_SDPA_MIN_GB", "12")) * (1024 ** 3))
_MAX_HEAD_DIM = 128  # kernel uses a thread-local acc[128]

_MSL = r"""
#include <metal_stdlib>
using namespace metal;

kernel void flash_attn_fwd(
    device const float* Q   [[buffer(0)]],   // [B,Hq,Lq,D]
    device const float* K   [[buffer(1)]],   // [B,Hkv,Lk,D]
    device const float* V   [[buffer(2)]],   // [B,Hkv,Lk,D]
    device float*       O   [[buffer(3)]],   // [B,Hq,Lq,D]
    device const int*   SH  [[buffer(4)]],   // [B,Hq,Hkv,Lq,Lk,D,causal]
    device const float* PR  [[buffer(5)]],   // [scale]
    uint3 gid [[thread_position_in_grid]])
{
    const int B=SH[0], Hq=SH[1], Hkv=SH[2], Lq=SH[3], Lk=SH[4], D=SH[5], causal=SH[6];
    const float scale = PR[0];

    const int qi = int(gid.x);
    const int bh = int(gid.y);
    if (qi >= Lq || bh >= B*Hq) return;
    const int b   = bh / Hq;
    const int hq  = bh % Hq;
    const int hkv = hq / (Hq / Hkv);            // grouped-query mapping

    const int q_base = ((b*Hq + hq)*Lq + qi)*D;
    const int kv_bh  = (b*Hkv + hkv);

    // causal aligns bottom-right (key j attends iff j <= qi + (Lk-Lq))
    const int kmax = causal ? (qi + (Lk - Lq) + 1) : Lk;

    float m = -INFINITY;
    float l = 0.0f;
    float acc[128];
    for (int d=0; d<D; ++d) acc[d]=0.0f;

    for (int kj=0; kj<kmax; ++kj) {
        const int k_base = (kv_bh*Lk + kj)*D;
        float s = 0.0f;
        for (int d=0; d<D; ++d) s += Q[q_base+d]*K[k_base+d];
        s *= scale;
        float m_new = max(m, s);
        float corr  = exp(m - m_new);
        float p     = exp(s - m_new);
        l = l*corr + p;
        for (int d=0; d<D; ++d) acc[d] = acc[d]*corr + p*V[k_base+d];
        m = m_new;
    }
    float inv = (l > 0.0f) ? (1.0f/l) : 0.0f;
    for (int d=0; d<D; ++d) O[q_base+d] = acc[d]*inv;
}
"""

_orig = None
_installed = False
_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_MSL)
    return _lib


def _flash(q, k, v, scale, is_causal):
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    qf = q.float().contiguous()
    kf = k.float().contiguous()
    vf = v.float().contiguous()
    out = torch.empty(B, Hq, Lq, D, device=q.device, dtype=torch.float32)
    sh = torch.tensor([B, Hq, Hkv, Lq, Lk, D, 1 if is_causal else 0],
                      dtype=torch.int32, device=q.device)
    pr = torch.tensor([float(scale)], dtype=torch.float32, device=q.device)
    _get_lib().flash_attn_fwd(qf, kf, vf, out, sh, pr,
                              threads=(Lq, B * Hq, 1), group_size=(64, 1, 1))
    return out.to(q.dtype)


_DEBUG = os.environ.get("APPLESILICON_FP8_SDPA_DEBUG", "").lower() in ("1", "true", "on")


def _eligibility(query, key, value, attn_mask, dropout_p):
    """Return (eligible, reason). reason names the disqualifying gate."""
    if query.device.type != "mps":
        return False, "not-mps"
    if attn_mask is not None:
        return False, "attn_mask"
    if dropout_p:
        return False, "dropout"
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False, f"ndim({query.dim()})"
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False, f"dtype({query.dtype})"
    D = query.shape[-1]
    if D > _MAX_HEAD_DIM or key.shape[-1] != D or value.shape[-1] != D:
        return False, f"head_dim({D})"
    Hq, Hkv = query.shape[1], key.shape[1]
    if Hkv == 0 or Hq % Hkv != 0:
        return False, f"heads({Hq}/{Hkv})"
    if key.shape[2] != value.shape[2]:
        return False, "kv-len-mismatch"
    Lq, Lk = query.shape[2], key.shape[2]
    score_bytes = query.shape[0] * query.shape[1] * Lq * Lk * 2
    if score_bytes < _MIN_SCORE_BYTES:
        return False, f"fits({score_bytes / 1024**3:.1f}GB)"
    return True, "flash"


def _eligible(query, key, value, attn_mask, dropout_p):
    return _eligibility(query, key, value, attn_mask, dropout_p)[0]


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
          is_causal=False, scale=None, **kwargs):
    eligible, reason = _eligibility(query, key, value, attn_mask, dropout_p)
    if _DEBUG:
        try:
            area = query.shape[-2] * key.shape[-2]
        except Exception:
            area = -1
        # Only log non-trivial attentions so the spam is bounded.
        if area >= (1 << 20) or eligible:
            sm_gb = (query.shape[0] * query.shape[1] * area * 2 / 1e9) if query.dim() == 4 else -1
            print(f"{TAG}[dbg] q={tuple(query.shape)} k={tuple(key.shape)} "
                  f"dt={str(query.dtype).split('.')[-1]} mask={attn_mask is not None} "
                  f"causal={is_causal} -> {reason} "
                  f"(would-be score≈{sm_gb:.2f} GB)", flush=True)
    if eligible:
        try:
            s = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
            return _flash(query, key, value, s, is_causal)
        except Exception as e:  # never crash — fall back to stock SDPA
            print(f"{TAG} flash kernel fell back ({e}); using stock SDPA")
    return _orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                 is_causal=is_causal, scale=scale, **kwargs)


def install():
    global _orig, _installed
    if _installed:
        return
    if os.environ.get("APPLESILICON_FP8_SDPA", "auto").lower() == "off":
        return
    _orig = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _sdpa
    torch.nn.functional.scaled_dot_product_attention = _sdpa
    _installed = True
    print(f"{TAG} F.scaled_dot_product_attention uses a tiled Metal flash kernel "
          f"on MPS only when the score matrix would exceed "
          f"{_MIN_SCORE_BYTES / 1024**3:.0f} GB; avoids the OOM "
          f"from materializing the score matrix.")
