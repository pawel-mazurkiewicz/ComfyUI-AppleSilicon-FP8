"""Fix: FP8 weight/bias cast crash in cast_bias_weight on MPS.

When a model is loaded with FP8 weights (fp8_e4m3fn / fp8_e5m2) and
manual_cast_dtype=bfloat16 (the standard UNETLoader "weight_dtype: fp8_e4m3fn"
path on a non-CUDA box), ComfyUI's manual_cast Linear/Conv layers store the
weight AND bias as raw FP8 tensors on the GPU and cast them up per forward.

On MPS that per-forward cast crashes.  cast_bias_weight does, for the plain
(non-vbar) path:

    weight = cast_to(s.weight, None, device, ...)   # dtype=None -> stays FP8
    bias   = cast_to(s.bias,   None, device, ...)   # dtype=None -> stays FP8
    ...
    bias   = bias.to(dtype=bias_dtype)              # ops.py ~371  FP8->bf16  ✗
    ...
    weight = weight.to(dtype=dtype)                 # ops.py ~376  FP8->bf16  ✗

MPS can *store* / move FP8 tensors but cannot cast TO or FROM them on-device,
so both .to() calls raise:

    TypeError: Trying to convert Float8_e4m3fn to the MPS backend but it does
               not have support for that dtype.

(The bias crashes first, which is why clamping bias_dtype alone wasn't enough —
the weight would have crashed on the very next line.)

Fix: for the plain FP8-on-MPS case we take over cast_bias_weight entirely and
decode weight/bias FP8 -> compute dtype via the LUT+gather path (decode_fp8 in
_common.py), which is MPS-safe and bit-exact.  weight_function / bias_function
(LoRA-as-function, etc.) are applied exactly as the original does.

Anything we don't recognise (vbar `_v` layers, QuantizedTensor weights, non-FP8
layers) is delegated back to the original implementation — with the historical
bias_dtype clamp kept as a belt-and-braces fallback for those paths.
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
        class _Never:  # isinstance(x, _Never) is always False
            pass
        return _Never


# Resolved at install() time.
_QuantizedTensor = None


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
        return True
    return param.dtype in FP8_DTYPES


def _to_compute(param, target_dtype, device):
    """Bring an FP8 / QuantizedTensor param to `target_dtype` on `device`, MPS-safe."""
    if param is None:
        return None
    if param.device != device:
        # Device move only — never a dtype cast, so FP8 survives the hop.
        param = param.to(device=device)
    if isinstance(param, _QuantizedTensor):
        # dequantize() routes through the comfy_kitchen eager path, which our
        # comfykitchen_fp8 patch already made MPS-safe.
        return param.dequantize().to(target_dtype)
    if param.dtype in FP8_DTYPES:
        return decode_fp8(param).to(target_dtype)
    return param.to(dtype=target_dtype)


def _fp8_safe_bias_dtype(bias_dtype, dtype, input_tensor):
    """Fallback for delegated paths: never let bias_dtype be FP8 on MPS."""
    if bias_dtype is not None:
        return torch.bfloat16 if bias_dtype in FP8_DTYPES else bias_dtype
    eff = _resolve_target_dtype(dtype, input_tensor)
    return eff  # already non-FP8 by construction


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

                w = _to_compute(weight, target, dev)
                for f in s.weight_function:
                    w = f(w)

                b = None
                if bias is not None:
                    b = _to_compute(bias, btarget, dev)
                    for f in s.bias_function:
                        b = f(b)

                if offloadable:
                    return (w, b, (None, None, None))
                return (w, b)

        # Delegate everything else; keep the bias_dtype clamp as a safety net.
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
