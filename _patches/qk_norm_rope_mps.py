"""Patch #29: fused per-head RMSNorm + interleaved RoPE (comfy_kitchen rms_rope / rms_rope1).

One compile_shader pass per tensor replaces eager's norm, rotate and layout launches: q (or k)
is read straight from its strided view of the fused QKV projection, fp32 norm * weight and the
rotation run in registers, with one rounding. The result is written contiguous in the input's
own logical layout, exactly what eager returns. Pure elementwise, so M1+ with no Metal-4.1
requirement. DEFAULT ON (ASFP8_QK_NORM_ROPE=off disables); each dtype is self-checked before
first use and any unsupported input takes the original eager function.
"""
from __future__ import annotations
import threading

import torch

TAG = "[AppleSilicon-FP8/qk-norm-rope]"

_MSL = r"""
#include <metal_stdlib>
using namespace metal;
constant constexpr uint TG = 64;
constant constexpr uint NSIMD = TG / 32;
kernel void qk_norm_rope(
    device const @T@*   x    [[buffer(0)]],
    device const @T@*   w    [[buffer(1)]],
    device const float* fr   [[buffer(2)]],
    device       @T@*   out  [[buffer(3)]],
    device const int*   meta [[buffer(4)]],
    device const float* epsb [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const int D = meta[0], H = meta[1], S = meta[2];
    const ulong sb = ulong(meta[3]), ss = ulong(meta[4]), sh = ulong(meta[5]);
    const int fb = meta[6];                       // table batch stride in tokens (0 or S)
    const int bhsd = meta[7];                     // 1: [B,H,S,D] in and out, 0: [B,S,H,D]
    const uint h = tgpig.y;
    const uint t = tgpig.z;                       // b*S + s
    const uint b = t / uint(S), s = t % uint(S);
    const ulong xbase = ulong(b) * sb + ulong(s) * ss + ulong(h) * sh;
    const ulong obase = (bhsd != 0)
        ? ((ulong(b) * ulong(H) + ulong(h)) * ulong(S) + ulong(s)) * ulong(D)
        : ((ulong(b) * ulong(S) + ulong(s)) * ulong(H) + ulong(h)) * ulong(D);
    const int halfD = D / 2;
    float local = 0.0f;
    for (uint p = lid; p < uint(halfD); p += TG) {
        const float a = float(x[xbase + 2u*p]);
        const float c = float(x[xbase + 2u*p + 1u]);
        local += a*a + c*c;
    }
    float warp = simd_sum(local);
    threadgroup float part[NSIMD];
    if (lane == 0) part[sgid] = warp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < NSIMD; ++i) tot += part[i];
    const float inv = rsqrt(tot / float(D) + epsb[0]);
    const ulong fbase = ((ulong(b) * ulong(fb) + ulong(s)) * ulong(halfD)) * 4ul;
    for (uint p = lid; p < uint(halfD); p += TG) {
        const ulong fo = fbase + ulong(p) * 4ul;
        const float a = float(x[xbase + 2u*p])      * inv * float(w[2u*p]);
        const float c = float(x[xbase + 2u*p + 1u]) * inv * float(w[2u*p + 1u]);
        out[obase + 2u*p]      = @T@(fr[fo + 0] * a + fr[fo + 1] * c);
        out[obase + 2u*p + 1u] = @T@(fr[fo + 2] * a + fr[fo + 3] * c);
    }
}
"""

_MSL_T = {torch.float16: "half", torch.bfloat16: "bfloat", torch.float32: "float"}
# self-check rel-RMS ceilings vs an fp32 reference (kernel error is ~1 ulp of the dtype)
_CHECK_TOL = {torch.float16: 3e-3, torch.bfloat16: 1.5e-2, torch.float32: 1e-4}
_TG = 64
_MAXZ = 65535                       # grid z is B*S tokens
_CACHE_MAX = 256
_libs: dict[str, object] = {}
_eps_cache: dict[tuple, torch.Tensor] = {}
_meta_cache: dict[tuple, torch.Tensor] = {}
_orig: dict = {}
_warned = False
_checked: dict = {}                 # dtype -> True (verified) | False (latched off)
_lock = threading.RLock()

_backend_events: list[tuple] = []   # (fn_name, "kernel"|"fallback", tuple(shape))
_MAX_EVENTS = 4096


def _record(name, backend, shape):
    _backend_events.append((name, backend, tuple(shape)))
    if len(_backend_events) > _MAX_EVENTS:
        del _backend_events[:-_MAX_EVENTS // 2]


def _last_backend():
    return _backend_events[-1][1] if _backend_events else "fallback"


def _get_lib(dtype):
    t = _MSL_T[dtype]
    lib = _libs.get(t)
    if lib is None:
        lib = torch.mps.compile_shader(_MSL.replace("@T@", t))
        _libs[t] = lib
    return lib


def _tensor_const(cache, key, vals, dtype, device):
    t = cache.get(key)
    if t is None:
        if len(cache) >= _CACHE_MAX:
            cache.clear()
        t = torch.tensor(vals, dtype=dtype, device=device)
        cache[key] = t
    return t


def _table(freqs_cis, halfD, S, B, layout_bshd):
    """(contiguous fp32 [Bf,S,halfD,4], table batch stride in tokens) or (None, 0)."""
    if freqs_cis.dtype != torch.float32 or freqs_cis.dim() != 6:
        return None, 0
    fs = freqs_cis.shape
    if fs[-1] != 2 or fs[-2] != 2 or fs[-3] != halfD:
        return None, 0
    if layout_bshd:                 # [Bf, S, 1, halfD, 2, 2]
        Bf, Lf, one = fs[0], fs[1], fs[2]
    else:                           # [Bf, 1, S, halfD, 2, 2]
        Bf, one, Lf = fs[0], fs[1], fs[2]
    if one != 1 or Lf != S or Bf not in (1, B):
        return None, 0
    return freqs_cis.reshape(Bf, Lf, halfD, 4).contiguous(), (0 if Bf == 1 else S)


def _launch(x, w, eps, fr, fb, layout_bshd):
    """x: 4-D MPS view. Returns a fresh contiguous tensor in x's logical layout."""
    if layout_bshd:
        B, S, H, D = x.shape
        sb, ss, sh = x.stride(0), x.stride(1), x.stride(2)
    else:
        B, H, S, D = x.shape
        sb, ss, sh = x.stride(0), x.stride(2), x.stride(1)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    meta = _tensor_const(_meta_cache, (D, H, S, sb, ss, sh, fb, layout_bshd, str(x.device)),
                         [D, H, S, sb, ss, sh, fb, 0 if layout_bshd else 1],
                         torch.int32, x.device)
    epsb = _tensor_const(_eps_cache, (float(eps), str(x.device)), [float(eps)],
                         torch.float32, x.device)
    _get_lib(x.dtype).qk_norm_rope(x, w, fr, out, meta, epsb,
                                   threads=(_TG, H, B * S), group_size=(_TG, 1, 1))
    return out


def _ok_tensor(x, scale, freqs_cis):
    if (x.device.type != "mps" or x.dtype not in _MSL_T or x.dim() != 4 or x.numel() == 0
            or scale is None or scale.dtype != x.dtype or scale.dim() != 1
            or scale.shape[0] != x.shape[-1] or scale.device != x.device
            or freqs_cis.device != x.device
            or x.shape[-1] % 2 != 0 or x.stride(-1) != 1 or not scale.is_contiguous()):
        return False
    return min(x.stride()) >= 0 and max(x.stride()) < 2 ** 31


def _layout(freqs_cis):
    """True: BSHD input ([B,S,1,..] table), False: BHSD ([B,1,S,..]), None: ambiguous."""
    if freqs_cis.dim() != 6:
        return None
    one_h, one_s = freqs_cis.shape[1] == 1, freqs_cis.shape[2] == 1
    if one_s and not one_h:
        return True
    if one_h and not one_s:
        return False
    return None


def _fused1(x, freqs_cis, scale, eps):
    """The fused result, or None to fall back."""
    if not _ok_tensor(x, scale, freqs_cis):
        return None
    bshd = _layout(freqs_cis)
    if bshd is None:
        return None
    B, S, H, D = x.shape if bshd else (x.shape[0], x.shape[2], x.shape[1], x.shape[3])
    if B * S > _MAXZ or H > _MAXZ:
        return None
    fr, fb = _table(freqs_cis, D // 2, S, B, bshd)
    if fr is None:
        return None
    return _launch(x, scale, eps, fr, fb, bshd)


def _reference(x, w, eps, fr6):
    """fp32 formula reference, independent of comfy_kitchen (self-check only)."""
    xf = x.float()
    n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    pr = n.reshape(*n.shape[:-1], -1, 1, 2)
    return (fr6[..., 0] * pr[..., 0] + fr6[..., 1] * pr[..., 1]).reshape(n.shape)


def _verify(dtype, device):
    """Synthetic BSHD (strided QKV view, batch table) and BHSD (shared table) launches."""
    g = torch.Generator().manual_seed(1234)
    B, S, H, D, eps = 2, 8, 3, 16, 1e-6
    w = (torch.rand(D, generator=g) + 0.5).to(dtype).to(device)

    def table(Bf, n):
        th = torch.rand(Bf, n, 1, D // 2, generator=g) * 3
        c, s = th.cos(), th.sin()
        return torch.stack([c, -s, s, c], -1).reshape(Bf, n, 1, D // 2, 2, 2).to(device)

    qkv = (torch.randn(B, S, 3 * H * D, generator=g) * 2).to(dtype).to(device)
    xs = qkv[..., H * D:2 * H * D].view(B, S, H, D)
    f = table(B, S)
    fr, fb = _table(f, D // 2, S, B, True)
    out = _launch(xs, w, eps, fr, fb, True)
    if out.shape != xs.shape or not out.is_contiguous():
        return False
    refs = [(out, _reference(xs, w, eps, f))]

    xh = (torch.randn(B, H, S, D, generator=g) * 2).to(dtype).to(device)
    f = table(1, S).movedim(1, 2)
    fr, fb = _table(f, D // 2, S, B, False)
    out = _launch(xh, w, eps, fr, fb, False)
    if out.shape != xh.shape or not out.is_contiguous():
        return False
    refs.append((out, _reference(xh, w, eps, f)))
    tol = _CHECK_TOL[dtype]
    for o, r in refs:
        if not bool(torch.isfinite(o.float()).all().item()):
            return False
        if ((o.float() - r).norm() / r.norm().clamp_min(1e-12)).item() >= tol:
            return False
    return True


def _latch_off(dtype, why):
    _checked[dtype] = False
    print(f"{TAG} {why} for {dtype}; fused qk-norm-rope off for this dtype.", flush=True)


def _dtype_ready(dtype, device):
    ok = _checked.get(dtype)
    if ok is not None:
        return ok
    with _lock:
        ok = _checked.get(dtype)
        if ok is not None:
            return ok
        try:
            ok = bool(_verify(dtype, device))
            if ok:
                _checked[dtype] = True
            else:
                _latch_off(dtype, "self-check failed")
        except Exception as e:
            ok = False
            _latch_off(dtype, f"self-check raised ({e!r})")
        return ok


def _bypass(x, scale, freqs_cis):
    if torch.compiler.is_compiling():
        return True
    return torch.is_grad_enabled() and any(
        getattr(t, "requires_grad", False) for t in (x, scale, freqs_cis))


def rms_rope1_fused(x, freqs_cis, scale, epsilon=1e-6):
    global _warned
    try:
        if (_checked.get(x.dtype) is not False and not _bypass(x, scale, freqs_cis)
                and _ok_tensor(x, scale, freqs_cis) and _dtype_ready(x.dtype, x.device)):
            out = _fused1(x, freqs_cis, scale, epsilon)
            if out is not None:
                _record("rms_rope1", "kernel", x.shape)
                return out
    except Exception as e:
        if not _warned:
            _warned = True
            print(f"{TAG} kernel error ({e}); falling back to eager.", flush=True)
        _checked[x.dtype] = False
    _record("rms_rope1", "fallback", x.shape)
    return _orig["rms_rope1"](x, freqs_cis, scale, epsilon)


def rms_rope_fused(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6):
    if k_scale is None:
        k_scale = q_scale
    return (rms_rope1_fused(q, freqs_cis, q_scale, epsilon),
            rms_rope1_fused(k, freqs_cis, k_scale, epsilon))


_PATCH_MAP = {"rms_rope1": "rms_rope1_fused", "rms_rope": "rms_rope_fused"}
_installed = False


def _targets():
    """The eager package the registry reads, and the eager.rope source module."""
    import comfy_kitchen.backends.eager as e
    import comfy_kitchen.backends.eager.rope as r
    return (e, r)


def _do_install():
    global _installed
    if _installed:
        return True
    e, r = _targets()
    orig = {name: getattr(e, name) for name in _PATCH_MAP}
    _orig.update(orig)
    g = globals()
    for name, repl in _PATCH_MAP.items():
        setattr(e, name, g[repl])
        try:
            setattr(r, name, g[repl])
        except Exception:
            pass
    _installed = True
    return True


def install():
    import os
    from . import _caps
    if os.environ.get("ASFP8_QK_NORM_ROPE", "").strip().lower() == "none":
        return
    if _installed or not _caps.resolve("ASFP8_QK_NORM_ROPE", default_on=True,
                                       cap=_caps.has_compile_shader):
        return
    try:
        _do_install()
    except Exception as e:
        print(f"{TAG} install failed ({e!r}); eager rms_rope untouched.", flush=True)
        return
    print(f"{TAG} fused QK-norm+RoPE active on MPS (comfy_kitchen rms_rope/rms_rope1 rerouted; "
          f"fp32 math, one rounding, per-dtype self-check).", flush=True)


def install_for_test():
    _do_install()


def uninstall_for_test():
    global _installed, _warned
    if not _installed:
        return
    e, r = _targets()
    for name, fn in _orig.items():
        setattr(e, name, fn)
        try:
            setattr(r, name, fn)
        except Exception:
            pass
    _orig.clear()
    _checked.clear()
    _warned = False
    _installed = False
