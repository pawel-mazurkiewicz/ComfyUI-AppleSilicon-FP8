"""Fix: torch.nn.functional.linear with FP8 inputs/weights on MPS.

`F.linear` is a C++ op. When called with FP8-dtype tensors on MPS it raises:

    RuntimeError: MPS device does not support linear for non-float weights
    RuntimeError: MPS device does not support linear for non-float inputs

This covers the gap that patch #8 (tensor_to_fp8.py / Tensor.to shim) cannot:
that patch intercepts Python-level `.to()` calls, but `F.linear` promotes dtypes
internally in C++, so no Python `.to()` is ever called.

Typical pattern (WanVideo T5 encoder, any custom Linear that stores fp8 weights):

    class FP8Linear(nn.Module):
        def forward(self, x):
            return F.linear(x, self.weight, self.bias)   # weight is fp8

We monkey-patch `torch.nn.functional.linear` so that, for MPS + FP8 operands:

  1. Decode FP8 input to the compute dtype (bf16 if input is FP8, else input.dtype).
  2. Decode FP8 weight to compute dtype.
  3. Decode FP8 bias (if present) to compute dtype.
  4. Call the original F.linear with the decoded tensors.

Non-MPS calls and calls with no FP8 operands take a tight fast path and hit the
original C++ kernel unchanged.
"""

import sys

import torch
import torch.nn.functional as F

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/linear_fp8]"

# Capture the original at import time so _patched_linear can be called directly
# in tests (without install()) and still have a valid reference to call through.
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
    # Fast path: no FP8 involved — hit the original kernel unchanged.
    if (input.dtype not in FP8_DTYPES
            and weight.dtype not in FP8_DTYPES
            and (bias is None or bias.dtype not in FP8_DTYPES)):
        return _original(input, weight, bias)

    # Fast path: MPS not in play — original handles it (or fails as before).
    if input.device.type != "mps":
        return _original(input, weight, bias)

    # Compute dtype: if the input itself is FP8 (rare), use bf16 so the matmul
    # lands on the matrix units; otherwise honour the input's existing dtype so
    # the decode is invisible to the rest of the pipeline.
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
    # Also patch torch.nn.functional directly on the module object so that
    # code that has already done `from torch.nn import functional as F` picks
    # up the patched version via the module attribute.
    torch.nn.functional.linear = _patched_linear
    _installed = True
    print(f"{TAG} F.linear FP8 operands decoded to compute dtype on MPS.")
