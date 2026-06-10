"""Fix: torch._scaled_mm with FP8 inputs on MPS (FLUX, SD3.5, etc.).

`torch._scaled_mm` is PyTorch's scaled matmul used for FP8 inference. MPS has no
kernel for it with FP8 operands, so FLUX / SD3.5 and similar FP8 models fail with:

    NotImplementedError: scaled_mm ... for MPS
    TypeError: ... convert Float8_e4m3fn to the MPS backend ...

We monkey-patch torch._scaled_mm so that, for MPS + FP8 operands, it:
  1. decodes both operands FP8 -> float32 via the LUT+gather (MPS-safe),
  2. applies the per-tensor / per-row scales,
  3. runs a native float32 matmul on MPS,
  4. applies bias / result-scale / out_dtype.

Non-MPS or non-FP8 calls fall through to the original implementation untouched.
This computes in float32 to avoid the FP16 overflow that raw FP8 dot products can
hit; it leans on MPS's native matmul rather than a custom Metal kernel, so there's
nothing to compile and no third-party code involved.
"""

import torch

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/scaled_mm]"

_original = None
_installed = False


def _to_f32(t):
    """FP8 -> float32 via LUT (MPS-safe); anything else -> float32 normally."""
    if t.dtype in FP8_DTYPES:
        return decode_fp8(t)
    return t.to(torch.float32)


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
            input,
            other,
            out_dtype=out_dtype,
            scale_a=scale_a,
            scale_b=scale_b,
            bias=bias,
            scale_result=scale_result,
            use_fast_accum=use_fast_accum,
        )

    # input: (M, K), other: (K, N) column-major — exactly torch._scaled_mm's layout.
    a = _to_f32(input)
    b = _to_f32(other)

    # Apply scales to the operands. Per-tensor scalars and per-row/col vectors all
    # broadcast over K: scale_a (M,1) over rows of a, scale_b (1,N) over cols of b.
    if scale_a is not None:
        a = a * scale_a.to(torch.float32)
    if scale_b is not None:
        b = b * scale_b.to(torch.float32)

    out = a @ b

    if bias is not None:
        out = out + bias.to(torch.float32)
    if scale_result is not None:
        out = out * scale_result.to(torch.float32)
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
    print(f"{TAG} torch._scaled_mm FP8 now runs on MPS via LUT decode + native matmul.")
