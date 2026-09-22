"""Fix: `F.linear` with FP8 inputs/weights on MPS (3rd-party fp8 Linears, T5).

Decodes the fp8 operands to the compute dtype first. Patch #8's Tensor.to shim cannot
cover this: F.linear promotes dtypes in C++, so no Python `.to()` is ever called.
"""

import sys

import torch
import torch.nn.functional as F

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/linear_fp8]"

# captured at import so _patched_linear works in tests without install()
_original = F.linear
_installed = False


def _decode_operand(t, compute_dtype):
    """Decode t to compute_dtype if it is FP8; otherwise cast to compute_dtype."""
    if t.dtype in FP8_DTYPES:
        return decode_fp8(t, compute_dtype)
    if t.dtype != compute_dtype:
        return t.to(compute_dtype)
    return t


def _patched_linear(input, weight, bias=None):
    if (input.dtype not in FP8_DTYPES
            and weight.dtype not in FP8_DTYPES
            and (bias is None or bias.dtype not in FP8_DTYPES)):
        return _original(input, weight, bias)

    if input.device.type != "mps":
        return _original(input, weight, bias)

    # bf16 for an fp8 input so the matmul lands on the matrix units; otherwise keep
    # the input's dtype so the decode is invisible downstream
    compute_dtype = torch.bfloat16 if input.dtype in FP8_DTYPES else input.dtype

    dec_input = _decode_operand(input, compute_dtype)
    dec_weight = _decode_operand(weight, compute_dtype)
    dec_bias = _decode_operand(bias, compute_dtype) if bias is not None else None

    return _original(dec_input, dec_weight, dec_bias)


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    F.linear = _patched_linear
    # also by module attribute, for code that imported functional under another name
    torch.nn.functional.linear = _patched_linear
    _installed = True
    print(f"{TAG} F.linear FP8 operands decoded to compute dtype on MPS.")
