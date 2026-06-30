"""Patch #18: fused single-pass RMSNorm + affine + adaLN(scale,shift) + residual on MPS.

out = residual + (rmsnorm(x) * weight) * (1 + scale) + shift   — one compile_shader kernel.

Bandwidth/launch win: replaces ~4-5 separate MPS elementwise/reduction launches (each a full
DRAM round-trip of the activation) with one kernel that reads x+residual once and writes out once,
fp32 accumulation throughout. One threadgroup per row, 128 threads, fp32 simd_sum reduction for
mean(x^2). The grid is z-tiled (row = z*ny + y, early-return) so dispatch is correct regardless of
any per-dimension threadgroup cap. 64-bit element offsets + fp32 math keep it correct at every row
count, so when installed it SUPERSEDES rmsnorm_mps_large.py's >2^21-row correctness fallback (which
only fixed the bare norm and reduced over all normalized dims).

Opt-in: ASFP8_FUSED_NORM=1. Never fatal; falls back to an exact, GROUP-AWARE torch composition on
any error, off-MPS, unsupported dtype, optional-tensor shape/device/dtype mismatch, or indivisible
modulation grouping. `_last_backend` records which path ran ("kernel" | "fallback") for tests.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

TAG = "[AppleSilicon-FP8/fused-norm]"

_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TG    = 128;
constant constexpr uint NSIMD = TG / 32;

kernel void fused_rmsnorm_modulate(
    device const @T@*   x        [[buffer(0)]],
    device const @T@*   weight   [[buffer(1)]],
    device const @T@*   scale    [[buffer(2)]],
    device const @T@*   shift    [[buffer(3)]],
    device const @T@*   residual [[buffer(4)]],
    device       @T@*   out      [[buffer(5)]],
    device const int*   meta     [[buffer(6)]],
    device const float* epsb     [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const int   D     = meta[0];
    const int   rpgS  = meta[1];
    const int   rpgH  = meta[2];
    const int   flags = meta[3];
    const uint  ny    = uint(meta[4]);
    const uint  rows  = uint(meta[5]);
    const float eps   = epsb[0];

    const uint  row  = tgpig.z * ny + tgpig.y;
    if (row >= rows) return;
    const ulong base = ulong(row) * ulong(D);

    float local = 0.0f;
    for (uint d = lid; d < uint(D); d += TG) {
        float xv = float(x[base + d]);
        local += xv * xv;
    }
    float warp = simd_sum(local);
    threadgroup float part[NSIMD];
    if (lane == 0) part[sgid] = warp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = (lane < NSIMD) ? part[lane] : 0.0f;
        part[0] = simd_sum(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = rsqrt(part[0] / float(D) + eps);

    const bool  hasS  = (flags & 1) != 0;
    const bool  hasH  = (flags & 2) != 0;
    const bool  hasR  = (flags & 4) != 0;
    const bool  hasW  = (flags & 8) != 0;
    const ulong sbase = hasS ? ulong(row / uint(rpgS)) * ulong(D) : 0ul;
    const ulong hbase = hasH ? ulong(row / uint(rpgH)) * ulong(D) : 0ul;

    for (uint d = lid; d < uint(D); d += TG) {
        float y = float(x[base + d]) * inv;
        if (hasW) y *= float(weight[d]);
        if (hasS) y *= (1.0f + float(scale[sbase + d]));
        if (hasH) y += float(shift[hbase + d]);
        if (hasR) y += float(residual[base + d]);
        out[base + d] = @T@(y);
    }
}
"""

_MSL_T = {torch.float16: "half", torch.bfloat16: "bfloat", torch.float32: "float"}
_TG = 128
_NY = 32768                      # threadgroups along grid.y per z-slab (z-tiling)
_libs: dict[str, object] = {}
_meta_cache: dict[tuple, torch.Tensor] = {}
_eps_cache: dict[tuple, torch.Tensor] = {}
_warned = False
_last_backend = "fallback"       # "kernel" | "fallback"; set before every return (test/bench spy)


def _get_lib(dtype):
    t = _MSL_T[dtype]
    lib = _libs.get(t)
    if lib is None:
        lib = torch.mps.compile_shader(_MSL.replace("@T@", t))
        _libs[t] = lib
    return lib


def _meta(D, rpgS, rpgH, flags, ny, rows, device):
    key = (int(D), int(rpgS), int(rpgH), int(flags), int(ny), int(rows), str(device))
    t = _meta_cache.get(key)
    if t is None:
        t = torch.tensor([D, rpgS, rpgH, flags, ny, rows], dtype=torch.int32, device=device)
        _meta_cache[key] = t
    return t


def _eps(eps, device):
    key = (float(eps), str(device))
    t = _eps_cache.get(key)
    if t is None:
        t = torch.tensor([float(eps)], dtype=torch.float32, device=device)
        _eps_cache[key] = t
    return t


def _expand_mod(t, D, rows):
    """Expand a modulation tensor [..., D] -> [rows, D] using the kernel's row->group map."""
    g = t.reshape(-1, D)
    G = g.shape[0]
    if G == rows:
        return g
    if rows % G == 0:
        return g.repeat_interleave(rows // G, dim=0)
    return g  # indivisible: caller's reshape raises a clear error


def _reference(x, weight, eps, scale, shift, residual):
    xf = x.float()
    D = xf.shape[-1]
    rows = xf.numel() // D
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    if scale is not None:
        out = out * (1.0 + _expand_mod(scale.float(), D, rows).reshape(out.shape))
    if shift is not None:
        out = out + _expand_mod(shift.float(), D, rows).reshape(out.shape)
    if residual is not None:
        out = out + residual.float()
    return out.to(x.dtype)


def _fallback(x, weight, eps, scale, shift, residual):
    global _last_backend
    _last_backend = "fallback"
    return _reference(x, weight, eps, scale, shift, residual)


def _group(t, D, rows):
    """Reshape a modulation tensor to [G, D] and return (contig_buffer, rows_per_group) or None."""
    g = t.contiguous().reshape(-1, D)
    G = g.shape[0]
    if rows % G != 0:
        return None
    return g, rows // G


def _ok_optional(t, x, *, exact_shape=None, numel=None):
    """Validate an optional tensor matches x.device/x.dtype and a shape/numel contract."""
    if t.device != x.device or t.dtype != x.dtype:
        return False
    if exact_shape is not None and tuple(t.shape) != tuple(exact_shape):
        return False
    if numel is not None and t.numel() != numel:
        return False
    return True


def fused_rmsnorm_modulate(x, weight, eps=1e-6, scale=None, shift=None, residual=None):
    global _last_backend
    if x.device.type != "mps" or x.dtype not in _MSL_T or x.numel() == 0:
        return _fallback(x, weight, eps, scale, shift, residual)
    try:
        D = x.shape[-1]
        xc = x.contiguous()
        rows = xc.numel() // D
        dummy = torch.empty(1, dtype=x.dtype, device=x.device)

        flags = 0
        wbuf = dummy
        if weight is not None:
            if not _ok_optional(weight, x, numel=D):
                return _fallback(x, weight, eps, scale, shift, residual)
            wbuf = weight.contiguous()
            flags |= 8
        sbuf, rpgS = dummy, 1
        if scale is not None:
            if scale.device != x.device or scale.dtype != x.dtype:
                return _fallback(x, weight, eps, scale, shift, residual)
            grp = _group(scale, D, rows)
            if grp is None:
                return _fallback(x, weight, eps, scale, shift, residual)
            sbuf, rpgS = grp[0], grp[1]
            flags |= 1
        hbuf, rpgH = dummy, 1
        if shift is not None:
            if shift.device != x.device or shift.dtype != x.dtype:
                return _fallback(x, weight, eps, scale, shift, residual)
            grp = _group(shift, D, rows)
            if grp is None:
                return _fallback(x, weight, eps, scale, shift, residual)
            hbuf, rpgH = grp[0], grp[1]
            flags |= 2
        rbuf = dummy
        if residual is not None:
            if not _ok_optional(residual, x, exact_shape=x.shape):
                return _fallback(x, weight, eps, scale, shift, residual)
            rbuf = residual.contiguous()
            flags |= 4

        ny = min(rows, _NY)
        nz = (rows + ny - 1) // ny
        out = torch.empty_like(xc)
        meta = _meta(D, rpgS, rpgH, flags, ny, rows, x.device)
        epsb = _eps(eps, x.device)
        _get_lib(x.dtype).fused_rmsnorm_modulate(
            xc, wbuf, sbuf, hbuf, rbuf, out, meta, epsb,
            threads=(_TG, ny, nz), group_size=(_TG, 1, 1),
        )
        _last_backend = "kernel"
        return out.view_as(x)
    except Exception as e:  # never fatal
        global _warned
        if not _warned:
            print(f"{TAG} kernel error ({e}); falling back to torch composition.", flush=True)
            _warned = True
        return _fallback(x, weight, eps, scale, shift, residual)


# ---- optional F.rms_norm reroute (supersedes rmsnorm_mps_large.py when installed) ----
_orig_rms_norm = None
_installed = False


def _rms_norm(input, normalized_shape, weight=None, eps=None):
    if input.device.type != "mps" or input.dtype not in _MSL_T:
        return _orig_rms_norm(input, normalized_shape, weight, eps)
    # Codex BLOCKER #2: reduce over ALL normalized dims -> flatten last len(normalized_shape) dims.
    D = 1
    for d in normalized_shape:
        D *= int(d)
    if D <= 0 or input.numel() % D != 0:
        return _orig_rms_norm(input, normalized_shape, weight, eps)
    e = eps if eps is not None else torch.finfo(input.dtype).eps
    x2d = input.contiguous().view(-1, D)
    w1d = weight.contiguous().view(D) if weight is not None else None
    out = fused_rmsnorm_modulate(x2d, w1d, e)
    return out.view_as(input)


def install():
    global _orig_rms_norm, _installed
    if _installed or os.environ.get("ASFP8_FUSED_NORM") != "1":
        return
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        return
    _orig_rms_norm = F.rms_norm
    F.rms_norm = _rms_norm
    torch.nn.functional.rms_norm = _rms_norm
    _installed = True
    print(f"{TAG} fused rmsnorm+modulation kernel active on MPS "
          f"(F.rms_norm rerouted; supersedes the >2^21-row fp32 fallback).", flush=True)


# ---- test-only install helpers (bypass the env flag so the reroute is exercised in pytest) ----
def install_for_test():
    global _orig_rms_norm, _installed
    if _installed:
        return
    _orig_rms_norm = F.rms_norm
    F.rms_norm = _rms_norm
    torch.nn.functional.rms_norm = _rms_norm
    _installed = True


def uninstall_for_test():
    global _orig_rms_norm, _installed
    if not _installed:
        return
    F.rms_norm = _orig_rms_norm
    torch.nn.functional.rms_norm = _orig_rms_norm
    _orig_rms_norm = None
    _installed = False
