"""Fix: tensor.to() FP8<->float conversions on MPS (third-party fp8 Linears).

ComfyUI's own layers go through comfy.ops (handled by ops_bias_fp8), but custom
nodes that roll their own fp8 Linear cast weights/bias at runtime with a plain
Python `.to()`, e.g. ComfyUI-WanVideoWrapper/custom_linear.py:

    weight = self.weight.to(input)        # fp8 weight -> input.dtype/device
    bias   = self.bias.to(input)          # fp8 bias   -> input.dtype/device

When the source is FP8 and the target dtype is float (or vice-versa) and MPS is
involved, this raises:

    TypeError: Trying to convert Float8_e4m3fn to the MPS backend but it does not
               have support for that dtype.

MPS can *store/move* FP8 tensors but cannot cast to/from FP8 on-device. We wrap
torch.Tensor.to so that, only when FP8 is actually involved and MPS is in play:

  * FP8 -> float : decode via the LUT+gather path (decode_fp8, MPS-safe), then
                   move to the requested device.
  * float -> FP8 : do the unsupported cast on CPU, then move the FP8 result to
                   the requested device (storage move is fine).

Everything else (the overwhelming common case) hits a tight fast path and calls
the original .to() unchanged. This only catches PYTHON-level .to() calls; FP8
type-promotion that happens inside C++ ops (e.g. F.linear with an fp8 weight)
is not visible here — see linear_fp8 / use a non-fp8 dtype for those layers.
"""

import sys

import torch

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/tensor_to]"

_FP8_SET = frozenset(FP8_DTYPES)
_installed = False


def _scan_target(args, kwargs, self_dtype):
    """Resolve (target_dtype, target_device) from .to() args without touching tensors unsafely."""
    target_dtype = kwargs.get("dtype")
    target_device = kwargs.get("device")
    for a in args:
        if isinstance(a, torch.dtype):
            target_dtype = a
        elif isinstance(a, torch.Tensor):
            if target_dtype is None:
                target_dtype = a.dtype
            if target_device is None:
                target_device = a.device
        elif isinstance(a, (torch.device, str, int)):
            target_device = a
    if target_dtype is None:
        target_dtype = self_dtype
    return target_dtype, target_device


def _target_has_fp8(args, kwargs):
    """Cheap check: is any explicit target dtype an FP8 type? (tensor-safe)"""
    kd = kwargs.get("dtype")
    if isinstance(kd, torch.dtype) and kd in _FP8_SET:
        return True
    for a in args:
        if isinstance(a, torch.dtype):
            if a in _FP8_SET:
                return True
        elif isinstance(a, torch.Tensor):
            if a.dtype in _FP8_SET:
                return True
    return False


def install():
    global _installed
    if _installed:
        return

    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    _orig_to = torch.Tensor.to

    def _patched_to(self, *args, **kwargs):
        self_fp8 = self.dtype in _FP8_SET

        # Fast path: source not FP8 and no FP8 target -> never our problem.
        if not self_fp8 and not _target_has_fp8(args, kwargs):
            return _orig_to(self, *args, **kwargs)

        target_dtype, target_device = _scan_target(args, kwargs, self.dtype)

        try:
            dev = torch.device(target_device) if target_device is not None else self.device
        except (TypeError, ValueError):
            dev = self.device

        if dev.type != "mps" and self.device.type != "mps":
            return _orig_to(self, *args, **kwargs)  # MPS not involved

        target_fp8 = target_dtype in _FP8_SET

        if self_fp8 and not target_fp8:
            # FP8 -> float: LUT decode on the source device, then move.
            out = decode_fp8(self).to(target_dtype)
            if out.device != dev:
                out = _orig_to(out, device=dev)
            return out

        if target_fp8 and not self_fp8:
            # float -> FP8: cast on CPU (unsupported on MPS), then move storage.
            src = self.detach()
            if src.device.type != "cpu":
                src = _orig_to(src, device="cpu")
            q = _orig_to(src, dtype=target_dtype)
            if dev.type != "cpu":
                q = _orig_to(q, device=dev)
            return q

        # Both FP8 (storage move) or neither FP8 (plain) -> original handles it.
        return _orig_to(self, *args, **kwargs)

    torch.Tensor.to = _patched_to
    _installed = True
    print(f"{TAG} torch.Tensor.to FP8<->float routed via LUT/CPU on MPS.")
