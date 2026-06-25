"""Fix: torch._scaled_mm with FP8 inputs on MPS (FLUX, SD3.5, etc.).

`torch._scaled_mm` is PyTorch's scaled matmul used for FP8 inference. MPS has no
kernel for it with FP8 operands, so FLUX / SD3.5 and similar FP8 models fail with:

    NotImplementedError: scaled_mm ... for MPS
    TypeError: ... convert Float8_e4m3fn to the MPS backend ...

We monkey-patch torch._scaled_mm so that, for MPS + FP8 operands, it:
  1. decodes both operands FP8 -> bf16 (bit-exact; bf16 has fp32 exponent range so
     no overflow) via the LUT+gather (MPS-safe),
  2. runs a bf16 matmul on MPS — bf16@bf16 dispatches to the matrix units
     (Neural Accelerators on M5+, simdgroup_matrix on M1–M4),
  3. applies per-row/col scales and bias to the output (in fp32 to avoid rounding),
  4. casts to out_dtype.

Non-MPS or non-FP8 calls fall through to the original implementation untouched.

EXPERIMENTAL (opt-in, ASFP8_FP8_EXT=1): for large fp8(e4m3) x fp8(e4m3) matmuls
this seam routes both operands through a Metal 4.1 fp8-native kernel (no bf16
decode), then applies scales/bias in fp32 — bit-exact vs the decode path, faster.
When the flag is unset/0 the fast path is inert and numerics are unchanged.
"""

import os

import torch

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/scaled_mm]"

_original = None
_installed = False

# fp8-native fast path (EXPERIMENTAL, opt-in via ASFP8_FP8_EXT=1). When the flag is
# unset/0 this whole path is inert and torch._scaled_mm behaves exactly as the decode
# implementation below — same numerics, zero extra work, no extension build.
_backend = None          # the cpp module, or False once known-unavailable
_self_checked = False


def _compute_dtype(out_dtype):
    """bf16 for bf16/fp16 results (rides the matrix units, bit-exact decode,
    fp32-range so no overflow); float32 only when the caller explicitly wants f32."""
    if out_dtype in (torch.bfloat16, torch.float16):
        return torch.bfloat16
    return torch.float32


def _decode(t, compute_dtype):
    if t.dtype in FP8_DTYPES:
        return decode_fp8(t, compute_dtype)
    return t.to(compute_dtype)


def _min_dim():
    """Weight-size threshold below which the fp8-native kernel is not worth its
    dispatch overhead; matches patch #15's knob so both share one tuning point."""
    try:
        return int(os.environ.get("ASFP8_FP8_EXT_MIN_DIM", "8192"))
    except ValueError:
        return 8192


def _fast_eligible(input, other):
    """Pure shape/dtype/env predicate (no device work, no import). Returns False
    instantly when ASFP8_FP8_EXT is unset/0 so the default decode path is untouched."""
    if os.environ.get("ASFP8_FP8_EXT") != "1":
        return False
    # The kernel decodes both operands as e4m3; only route when that's exact.
    if input.dtype != torch.float8_e4m3fn or other.dtype != torch.float8_e4m3fn:
        return False
    if input.device.type != "mps" or other.device.type != "mps":
        return False
    if input.dim() != 2 or other.dim() != 2:
        return False
    # input [M,K] @ other [K,N]; weight recovered as other.t() = [N,K].
    K = int(input.shape[1])
    N = int(other.shape[1])
    return max(N, K) >= _min_dim()


def _get_backend():
    """Lazily build + parity-self-check the fp8 extension. Returns the module or None
    (permanently, after a failure) so the caller falls back to the decode path."""
    global _backend, _self_checked
    if _backend is not None:
        return _backend or None
    from .fp8_ext.loader import module
    mod = module()
    if mod is None:
        _backend = False
        return None
    if not _self_checked:
        _self_checked = True
        try:
            torch.manual_seed(0)
            a = (torch.randn(64, 8192) * 0.3).to(torch.float8_e4m3fn)
            w = (torch.randn(8192, 8192) * 0.3).to(torch.float8_e4m3fn)   # [N,K]
            a_u8 = a.view(torch.uint8).to("mps").contiguous()
            w_u8 = w.view(torch.uint8).to("mps").contiguous()
            ref = decode_fp8(a.to("mps"), torch.float32) @ decode_fp8(w.to("mps"), torch.float32).t()
            raw = mod.fp8fp8_matmul2d_nt(a_u8, w_u8, 8192, 8192)
            rel = ((raw - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
            if rel >= 5e-2:
                print(f"{TAG} fp8-native self-check failed (rel={rel:.4f}); using decode path.")
                _backend = False
                return None
        except Exception as e:
            print(f"{TAG} fp8-native self-check raised; using decode path: {e!r}")
            _backend = False
            return None
    _backend = mod
    print(f"{TAG} fp8-native scaled_mm enabled (Metal 4.1 ext; min_dim={_min_dim()}).")
    return _backend


def _fast_route(input, other, scale_a, scale_b, scale_result, bias, out_dtype):
    """fp8xfp8 -> f32 via the Metal 4.1 kernel, then scales/bias in f32. Equivalent to
    the decode path (bench: bit-exact) but skips two fp8->bf16 decodes. Raises on any
    failure so the caller delegates to the decode implementation."""
    mod = _get_backend()
    if mod is None:
        raise RuntimeError("fp8 extension unavailable")
    K = int(input.shape[1])
    N = int(other.shape[1])
    a_u8 = input.contiguous().view(torch.uint8)
    # other is [K,N] (typically W.t()); the kernel wants W=[N,K] contiguous fp8 bytes.
    w_u8 = other.t().contiguous().view(torch.uint8)
    out = mod.fp8fp8_matmul2d_nt(a_u8, w_u8, K, N)   # [M,N] f32, unscaled
    if scale_a is not None:
        out = out * scale_a.to(torch.float32)
    if scale_b is not None:
        out = out * scale_b.to(torch.float32)
    if scale_result is not None:
        out = out * scale_result.to(torch.float32)
    if bias is not None:
        out = out + bias.to(torch.float32)
    if out_dtype is not None:
        out = out.to(out_dtype)
    return out


def _mps_scaled_mm(
    input,
    other,
    *,
    out_dtype=None,
    scale_a=None,
    scale_b=None,
    bias=None,
    scale_result=None,
    use_fast_accum=False,
):
    is_mps = input.device.type == "mps"
    is_fp8 = input.dtype in FP8_DTYPES or other.dtype in FP8_DTYPES
    if not (is_mps and is_fp8):
        return _original(
            input, other,
            out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b,
            bias=bias, scale_result=scale_result, use_fast_accum=use_fast_accum,
        )

    # EXPERIMENTAL opt-in fast path: fp8xfp8 -> f32 via the Metal 4.1 kernel. Any
    # failure (build/parity/runtime) falls through to the decode path below, so a
    # render never breaks. Inert unless ASFP8_FP8_EXT=1.
    if _fast_eligible(input, other):
        try:
            return _fast_route(input, other, scale_a, scale_b, scale_result, bias, out_dtype)
        except Exception as e:
            print(f"{TAG} fp8-native path failed ({e!r}); delegating to decode path.")

    compute_dtype = _compute_dtype(out_dtype)

    # input: (M,K), other: (K,N) column-major — torch._scaled_mm's layout.
    a = _decode(input, compute_dtype)
    b = _decode(other, compute_dtype)

    # bf16@bf16 -> bf16 (fp32 accumulate) on the matrix units; f32@f32 -> f32.
    out = a @ b

    # Per-row/col and per-tensor scales factor out of the dot product, so apply
    # them to the result (scale_a over rows, scale_b over cols). Compute in fp32
    # then come back to the working dtype to avoid intermediate rounding.
    if scale_a is not None or scale_b is not None or scale_result is not None:
        acc = out.to(torch.float32)
        if scale_a is not None:
            acc = acc * scale_a.to(torch.float32)
        if scale_b is not None:
            acc = acc * scale_b.to(torch.float32)
        if scale_result is not None:
            acc = acc * scale_result.to(torch.float32)
        out = acc.to(out.dtype)

    if bias is not None:
        # Always add bias in f32 regardless of compute_dtype so that bf16
        # compute does not lose precision in the bias term before the final
        # widen (matches the old unconditional f32 bias behaviour).
        out = out.to(torch.float32) + bias.to(torch.float32)
    if out_dtype is not None:
        out = out.to(out_dtype)
    return out


def install():
    global _original, _installed
    if _installed:
        return
    if not hasattr(torch, "_scaled_mm"):
        return  # requires PyTorch 2.4+
    _original = torch._scaled_mm
    torch._scaled_mm = _mps_scaled_mm
    _installed = True
    print(f"{TAG} torch._scaled_mm FP8 on MPS via LUT decode + bf16 matrix-unit matmul.")
