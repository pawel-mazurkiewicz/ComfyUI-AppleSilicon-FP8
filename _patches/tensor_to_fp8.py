"""Fix: tensor.to() FP8<->float conversions on MPS (third-party fp8 Linears).

MPS can store and move fp8 tensors but not cast them, so wrap torch.Tensor.to: fp8 ->
float decodes through the LUT, float -> fp8 casts on CPU and moves the storage back.
Only catches Python-level .to(); fp8 promotion inside C++ ops belongs to linear_fp8.
"""

import sys

import torch

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/tensor_to]"

_FP8_SET = frozenset(FP8_DTYPES)
_installed = False

# `.float()` and friends bind straight to their own aten ops, so wrapping
# torch.Tensor.to leaves them unpatched. `.double()` is deliberately absent: MPS has no
# float64 to rescue a result into, so it must keep raising torch's own message.
_DTYPE_SHORTCUTS = {
    "float": torch.float32,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
}


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
    """Is any explicit target dtype an FP8 type? Cheap, and never touches tensor data."""
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
            # fp8 -> float: LUT decode on the source device, then move
            out = decode_fp8(self).to(target_dtype)
            if out.device != dev:
                out = _orig_to(out, device=dev)
            return out

        if target_fp8 and not self_fp8:
            # float -> fp8: cast on CPU, then move the storage
            src = self.detach()
            if src.device.type != "cpu":
                src = _orig_to(src, device="cpu")
            q = _orig_to(src, dtype=target_dtype)
            if dev.type != "cpu":
                q = _orig_to(q, device=dev)
            return q

        # both fp8 (a storage move) or neither: the original handles it
        return _orig_to(self, *args, **kwargs)

    torch.Tensor.to = _patched_to

    def _make_shortcut(orig, dtype):
        # no *args: memory_format is keyword-only here, so a positional call must stay
        # a TypeError instead of reaching .to(), whose second positional is non_blocking
        def _patched(self, **kwargs):
            if self.dtype in _FP8_SET and self.device.type == "mps":
                return decode_fp8(self).to(dtype, **kwargs)
            return orig(self, **kwargs)
        return _patched

    for _name, _dtype in _DTYPE_SHORTCUTS.items():
        _orig = getattr(torch.Tensor, _name)
        setattr(torch.Tensor, _name, _make_shortcut(_orig, _dtype))

    _installed = True
    print(f"{TAG} torch.Tensor.to (+ .float()/.half()/.bfloat16()) "
          f"FP8<->float routed via LUT/CPU on MPS.")
