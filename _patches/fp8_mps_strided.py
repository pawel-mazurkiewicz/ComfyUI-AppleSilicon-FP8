"""Patch: strided / non-contiguous FP8 tensor ops on MPS (the global fp8 copy fix).

MPS has no strided float8 copy kernel, so materializing a non-contiguous fp8 tensor
raises "Undefined type Float8_e4m3fn". Wrap the primitive rather than each call site.
"""

import sys

import torch

TAG = "[AppleSilicon-FP8/fp8_strided]"

_installed = False
_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)
# materializing methods only: views never copy, so they don't crash
_METHODS = ("reshape", "contiguous", "clone")


def _wrap(orig):
    def method(self, *args, **kwargs):
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
