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
"""

import torch

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/scaled_mm]"

_original = None
_installed = False


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
        out = out + bias.to(out.dtype)
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
