"""Fix: INT8-quantized models silently run on CPU (and appear to hang) on MPS.

INT8 nodes (e.g. the `int8-fast` custom node used by Krea2 "int8_mixed"
checkpoints) quantize activations per-row and do the matmul with the integer
GEMM `torch._int_mm` (int8 x int8 -> int32):

    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)            # <- no MPS kernel
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)

PyTorch has no Metal kernel for `aten::_int_mm`. With PYTORCH_ENABLE_MPS_FALLBACK
on (ComfyUI sets it on Mac), the call doesn't crash — it silently copies both
operands MPS->CPU, runs the matmul on the CPU, and copies the int32 result back.
Per Linear, per layer, per sampling step. On a 12 GB Krea2/FLUX model that is
hundreds of CPU round-trips per step, so generation sits at 0/N with the GPU
idle — looks like a hang. (Without the fallback it would instead raise
"not implemented for MPS".)

Triton is the other code path these nodes ship, but Triton has no Metal backend
and usually isn't installed on a Mac, so the pure-PyTorch `_int_mm` path above is
what actually runs.

Fix: on MPS, do the integer matmul on the GPU in float32 instead of bouncing to
the CPU. int8 x int8 partial sums stay far under int32 range (|val| <= 127*127*K,
~2.5e8 for the widest FLUX layers, vs int32's 2.1e9), and the result is
immediately rescaled to bf16 downstream, so float32's ~6e-8 relative error is far
below the final bf16 precision. int8->float32 casts natively on MPS, so the whole
op stays on the GPU. Non-MPS devices keep the exact integer kernel untouched.
"""

import sys

import torch

TAG = "[AppleSilicon-FP8/int_mm]"

_orig = None
_installed = False


def _int_mm_mps(a, b):
    # Only intervene when an operand is on MPS; everything else keeps the exact
    # native integer kernel (CUDA/CPU).
    if a.device.type == "mps" or b.device.type == "mps":
        out = torch.mm(a.to(torch.float32), b.to(torch.float32))
        # Sums are integer-valued; round before narrowing so float rounding noise
        # can't truncate e.g. 14.9999997 -> 14. Safe: |out| << 2**31.
        return out.round_().to(torch.int32)
    return _orig(a, b)


def install():
    global _orig, _installed
    if _installed:
        return

    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    if not hasattr(torch, "_int_mm"):
        return

    _orig = torch._int_mm
    torch._int_mm = _int_mm_mps
    _installed = True
    print(f"{TAG} torch._int_mm runs on GPU (float32) on MPS instead of falling back to CPU (INT8 models).")
