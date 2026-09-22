"""Fix: torch._int_mm has no Metal kernel, so INT8 models silently run it on the CPU.

Does the matmul on the GPU in float32 instead. int8 x int8 sums stay far under int32
range and the result is rescaled to bf16 downstream, so float32 error never shows.
"""

import sys

import torch

TAG = "[AppleSilicon-FP8/int_mm]"

_orig = None
_installed = False


def _int_mm_mps(a, b):
    # non-MPS keeps the exact native integer kernel
    if a.device.type == "mps" or b.device.type == "mps":
        out = torch.mm(a.to(torch.float32), b.to(torch.float32))
        # round before narrowing: float noise would truncate 14.9999997 -> 14
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
