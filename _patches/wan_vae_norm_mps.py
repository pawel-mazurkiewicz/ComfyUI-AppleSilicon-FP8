"""Patch: single-pass channel RMS_norm for the Wan 2.1/2.2 VAE on MPS.

Stock is F.normalize(x, dim=1) * scale * gamma + bias, 5-6 passes; one compile_shader
kernel does it in one (fp32 accumulate), verified against the stock forward before first use.
DEFAULT ON where compile_shader works; ASFP8_WAN_VAE_NORM=off disables.
Ported from Illuminus7/ComfyUI-AppleSilicon-int8-Fast (MIT).
"""
from __future__ import annotations

import sys

import torch

from . import _caps

TAG = "[AppleSilicon-FP8/wan_vae_norm]"
NAME = "wan_vae_norm"

# one thread per 4 spatial positions; x, out are [B, C, S] contiguous
_MSL = r"""
#include <metal_stdlib>
using namespace metal;

kernel void channel_norm(const device @T@* x [[buffer(0)]], device @T@* out [[buffer(1)]],
                         const device float* gamma [[buffer(2)]], const device float* bias [[buffer(3)]],
                         constant uint& C [[buffer(4)]], constant uint& S [[buffer(5)]],
                         constant float& scale [[buffer(6)]], constant uint& has_bias [[buffer(7)]],
                         uint i [[thread_position_in_grid]]) {
  const uint b = i / (S / 4), s = (i % (S / 4)) * 4;
  const ulong base = ulong(b) * C * S + s;
  float4 ss = 0.0f;
  for (uint c = 0; c < C; ++c) {
    const float4 v = float4(*(const device vec<@T@, 4>*)(x + base + ulong(c) * S));
    ss += v * v;
  }
  const float4 r = scale / max(sqrt(ss), 1e-12f);
  for (uint c = 0; c < C; ++c) {
    float4 v = float4(*(const device vec<@T@, 4>*)(x + base + ulong(c) * S)) * r * gamma[c];
    if (has_bias) v += bias[c];
    *(device vec<@T@, 4>*)(out + base + ulong(c) * S) = vec<@T@, 4>(v);
  }
}
"""

_MSL_T = {torch.bfloat16: "bfloat", torch.float16: "half", torch.float32: "float"}
# max relative L2 error vs the fp32 stock forward, ~10x the output-rounding error of each dtype
_TOL = {torch.bfloat16: 1e-2, torch.float16: 2e-3, torch.float32: 1e-5}

_libs: dict = {}   # dtype -> shader library that compiled and passed verification
_orig_forward = None
_installed = False


def _sig(t):
    # _version raises on inference-mode tensors; the caller then skips the cache
    return (t.device, t.data_ptr(), t._version, t.dtype)


def _affine(self, x):
    """fp32 flat gamma/bias on x's device, cached on the module; None if not stock-broadcastable."""
    g, b = self.gamma, self.bias
    want = (x.shape[1],) + (1,) * (x.dim() - 2)
    if tuple(g.shape) != want or (b is not None and tuple(b.shape) != want):
        return None
    try:
        key = (x.device, _sig(g), None if b is None else _sig(b))
    except RuntimeError:
        key = None
    hit = self.__dict__.get("_asfp8_wan_norm")
    if key is not None and hit is not None and hit[0] == key:
        return hit[1], hit[2]

    def flat(t):
        return t.detach().to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()

    out = flat(g), None if b is None else flat(b)
    if key is not None:
        self.__dict__["_asfp8_wan_norm"] = (key, *out)
    return out


def _channel_norm(lib, x, gamma, bias, scale):
    B, C = x.shape[0], x.shape[1]
    S = x.numel() // (B * C)
    x = x.contiguous()
    out = torch.empty_like(x)
    lib.channel_norm(x, out, gamma, gamma if bias is None else bias, C, S, float(scale),
                     int(bias is not None), threads=(B * S // 4, 1, 1))
    return out


def _verify():
    from comfy.ldm.wan.vae import RMS_norm
    stock = _orig_forward or RMS_norm.forward
    gen = torch.Generator().manual_seed(0)
    _libs.clear()
    for dt, tol in _TOL.items():
        try:
            lib = torch.mps.compile_shader(_MSL.replace("@T@", _MSL_T[dt]))
        except Exception as e:
            print(f"{TAG} {dt} shader did not compile; stock RMS_norm for it: {e!r}", flush=True)
            continue
        worst = 0.0
        for images, bias, shape in ((False, True, (2, 40, 1, 6, 10)), (True, False, (2, 40, 6, 10))):
            norm = RMS_norm(40, images=images, bias=bias)
            with torch.no_grad():
                norm.gamma.copy_(torch.randn(norm.gamma.shape, generator=gen))
                if bias:
                    norm.bias.copy_(torch.randn(norm.bias.shape, generator=gen))
                x = torch.randn(shape, generator=gen).to("mps", dt)
                ref = stock(norm, x.cpu().float())
                a = _affine(norm, x)
                if a is None:
                    worst = float("inf")
                    break
                out = _channel_norm(lib, x, a[0], a[1], norm.scale).cpu().float()
            worst = max(worst, ((out - ref).norm() / ref.norm()).item())
        if worst < tol:
            _libs[dt] = lib
        else:
            print(f"{TAG} {dt} self-check mismatch (rel err {worst:.3g}); stock RMS_norm for it.", flush=True)
    return bool(_libs)


def _eligible(self, x):
    if not (self.channel_first and x.device.type == "mps" and x.dtype in _MSL_T
            and x.dim() >= 3 and x.numel() > 0):
        return False
    if torch.is_grad_enabled() and (x.requires_grad or self.gamma.requires_grad
                                    or (self.bias is not None and self.bias.requires_grad)):
        return False
    # vec4 access: S % 4 == 0 and a 4-element-aligned start
    if (x.numel() // (x.shape[0] * x.shape[1])) % 4 or x.storage_offset() % 4:
        return False
    return _caps.kernel_ready(NAME, _verify) and x.dtype in _libs


def _forward(self, x):
    try:
        if _eligible(self, x):
            a = _affine(self, x)
            if a is not None:
                return _channel_norm(_libs[x.dtype], x, a[0], a[1], self.scale)
    except Exception as e:  # never break a render; every later call goes to stock
        _caps.mark_kernel_failed(NAME)
        print(f"{TAG} kernel failed ({e!r}); using the stock RMS_norm from here on.", flush=True)
    return _orig_forward(self, x)


def install():
    global _orig_forward, _installed
    if _installed or sys.platform != "darwin" or not _caps.has_compile_shader():
        return
    if not _caps.resolve("ASFP8_WAN_VAE_NORM", default_on=True, cap=_caps.has_compile_shader):
        return
    try:
        from comfy.ldm.wan.vae import RMS_norm
        orig = RMS_norm.forward
    except Exception:
        return
    if not callable(orig) or orig is _forward:
        return
    # commit only after every lookup succeeded
    _orig_forward = orig
    RMS_norm.forward = _forward
    _installed = True
    print(f"{TAG} Wan VAE channel RMS_norm fused on MPS (self-check on first use).", flush=True)
