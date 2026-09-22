"""Fix: FP8 weight/bias cast crash in cast_bias_weight on MPS (fp8 UNETLoader).

MPS can store and move fp8 tensors but not cast them, so both of cast_bias_weight's
per-forward `.to()` calls raise. Take the plain fp8 path over and LUT-decode instead;
anything else (vbar `_v` layers, non-fp8 QuantizedTensors) delegates to the original
with the bias_dtype clamp as a safety net.
"""

import sys

import torch

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/ops_bias]"

_installed = False


def _get_quantized_tensor_cls():
    """Real QuantizedTensor class (or a never-matching sentinel if unavailable)."""
    try:
        from comfy.quant_ops import QuantizedTensor
        return QuantizedTensor
    except Exception:
        class _Never:  # isinstance() against this is always False
            pass
        return _Never


_QuantizedTensor = None   # resolved in install()


def _effective_device(device, input_tensor):
    if device is not None:
        return device
    if input_tensor is not None:
        return input_tensor.device
    return None


def _resolve_target_dtype(dtype, input_tensor):
    """Compute dtype the weight should end up in (mirrors cast_bias_weight top)."""
    target = dtype
    if target is None and input_tensor is not None:
        params = getattr(input_tensor, "params", None)
        if params is not None:
            target = getattr(params, "orig_dtype", None)
        if target is None:
            target = input_tensor.dtype
    if target is None or target in FP8_DTYPES:
        return torch.bfloat16
    return target


def _needs_handling(param):
    """True if param can't be cast off FP8 by a plain .to() on MPS."""
    if param is None:
        return False
    if isinstance(param, _QuantizedTensor):
        # only fp8 storage needs rescuing: int8/int4 layouts must keep their wrapper,
        # because comfy reaches into it for the raw storage
        return getattr(param, "storage_dtype", None) in FP8_DTYPES
    return param.dtype in FP8_DTYPES


def _to_compute(param, target_dtype, device):
    """Bring an FP8 / QuantizedTensor param to `target_dtype` on `device`, MPS-safe."""
    if param is None:
        return None
    if param.device != device:
        param = param.to(device=device)   # device only, so fp8 survives the hop
    if isinstance(param, _QuantizedTensor):
        # dequantize() routes through the eager path comfykitchen_fp8 made MPS-safe
        return param.dequantize().to(target_dtype)
    if param.dtype in FP8_DTYPES:
        return decode_fp8(param).to(target_dtype)
    return param.to(dtype=target_dtype)


def _bring(param, target_dtype, device):
    """Rescue only the param that needs it.

    The fast path is chosen per layer, so an fp8 bias can pull in a weight that was
    fine as-is, and dequantizing that one would strip a wrapper comfy still needs.
    """
    if param is None:
        return None
    if _needs_handling(param):
        return _to_compute(param, target_dtype, device)
    # a passed-through QuantizedTensor reaches weight_function still wrapped, where
    # native would have dequantized it first
    if param.device != device:
        param = param.to(device=device)
    if param.dtype == target_dtype:
        return param
    return param.to(dtype=target_dtype)


def _fp8_safe_bias_dtype(bias_dtype, dtype, input_tensor):
    """Fallback for delegated paths: never let bias_dtype be FP8 on MPS."""
    if bias_dtype is not None:
        return torch.bfloat16 if bias_dtype in FP8_DTYPES else bias_dtype
    eff = _resolve_target_dtype(dtype, input_tensor)
    return eff  # non-fp8 by construction


def install():
    global _installed, _QuantizedTensor
    if _installed:
        return

    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    try:
        import comfy.ops as ops_mod
    except ImportError:
        return

    if not hasattr(ops_mod, "cast_bias_weight"):
        return

    _QuantizedTensor = _get_quantized_tensor_cls()

    _original = ops_mod.cast_bias_weight

    def _mps_safe_cast_bias_weight(
        s,
        input=None,
        dtype=None,
        device=None,
        bias_dtype=None,
        offloadable=False,
        compute_dtype=None,
        want_requant=False,
    ):
        dev = _effective_device(device, input)
        dev_type = getattr(dev, "type", None)

        if dev_type == "mps" and not hasattr(s, "_v"):
            weight = getattr(s, "weight", None)
            bias = getattr(s, "bias", None)
            if _needs_handling(weight) or _needs_handling(bias):
                target = _resolve_target_dtype(dtype, input)
                btarget = target
                if bias_dtype is not None and bias_dtype not in FP8_DTYPES:
                    btarget = bias_dtype

                w = _bring(weight, target, dev)
                for f in s.weight_function:
                    w = f(w)

                b = None
                if bias is not None:
                    b = _bring(bias, btarget, dev)
                    for f in s.bias_function:
                        b = f(b)

                if offloadable:
                    return (w, b, (None, None, None))
                return (w, b)

        # delegate the rest, keeping the bias_dtype clamp as a safety net
        if dev_type == "mps":
            bias_dtype = _fp8_safe_bias_dtype(bias_dtype, dtype, input)

        return _original(
            s,
            input=input,
            dtype=dtype,
            device=device,
            bias_dtype=bias_dtype,
            offloadable=offloadable,
            compute_dtype=compute_dtype,
            want_requant=want_requant,
        )

    ops_mod.cast_bias_weight = _mps_safe_cast_bias_weight
    _installed = True
    print(f"{TAG} cast_bias_weight FP8 weight+bias LUT-decoded to compute dtype on MPS.")
