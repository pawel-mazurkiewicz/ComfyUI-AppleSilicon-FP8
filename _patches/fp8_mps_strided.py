"""Patch: make strided / non-contiguous FP8 tensor ops work on MPS (the global fp8 copy fix).

MPS has no copy kernel for float8 with non-trivial strides. Materializing a
non-contiguous fp8 tensor — `reshape` after a `transpose`, `.contiguous()` on a strided
view, `clone()` of a non-contiguous fp8 — raises:

    RuntimeError: Undefined type Float8_e4m3fn

Plain views/transposes and contiguous->contiguous copies are fine; only the *strided
materialization* fails. (The separate fp8<->float *dtype cast* limitation — "Trying to
convert Float8_e4m3fn to the MPS backend" — is handled by the Tensor.to patch.)

Comfy's quant/dequant code hits this in many places and the list keeps growing:
  comfy/float.py            to_blocked / from_blocked        (NVFP4 quantize)
  comfy_kitchen/float_utils from_blocked                     (NVFP4/MXFP8 dequantize)
  ... and any future microscaling layout that swizzles fp8 block-scales.

Rather than patch each call site, we fix the primitive once: wrap the materializing
Tensor methods so that, *only* for fp8 tensors on MPS, we attempt the op natively and —
if MPS rejects it with the "Float8" error — redo it on the CPU and move the (still-fp8)
result back. This is bit-exact (pure data rearrangement; .cpu() handles strided fp8
fine) and a no-op for every non-fp8 tensor: the guard short-circuits on `dtype` before
touching anything, so the hot path just falls through to the original method.

Disable with ASFP8_DISABLE=fp8_mps_strided if it ever needs to be ruled out.
"""

import sys

import torch

TAG = "[AppleSilicon-FP8/fp8_strided]"

_installed = False
_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)
# Materializing methods that trigger the strided-fp8 copy. Views (view/transpose/
# permute/expand) never copy, so they don't crash and aren't wrapped.
_METHODS = ("reshape", "contiguous", "clone")


def _wrap(orig):
    def method(self, *args, **kwargs):
        # Cheap guard first: non-fp8 tensors (the overwhelming majority) fall straight
        # through with just a dtype check.
        if self.dtype in _FP8 and self.device.type == "mps":
            try:
                return orig(self, *args, **kwargs)
            except RuntimeError as e:
                if "Float8" not in str(e):
                    raise  # a real error, not the MPS strided-fp8 limitation
                return orig(self.cpu(), *args, **kwargs).to("mps")
        return orig(self, *args, **kwargs)

    return method


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    for name in _METHODS:
        setattr(torch.Tensor, name, _wrap(getattr(torch.Tensor, name)))
    _installed = True
    print(f"{TAG} strided FP8 ops ({', '.join(_METHODS)}) fall back to CPU on MPS (global fp8 copy fix).")
