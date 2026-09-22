"""Fix: comfy_kitchen FP8 quantization on MPS (e.g. Ideogram 4).

The eager backend dequantizes with plain fp8 casts, which MPS rejects; replace them
with the LUT decode, and reroute the NVFP4/MXFP8 block-scale swizzles through the CPU.
The registry resolves implementations by getattr() at call time, so overwriting the
module attributes is picked up by every dispatch.
"""

import sys

import torch

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/comfy_kitchen]"

_installed = False


def _cpu_dequant_on_mps(orig):
    """Run a comfy_kitchen eager dequant on CPU when its inputs are on MPS, then move
    the float result back. The block-scales are tiny, so the round-trip is cheap."""
    def wrapped(*args, **kwargs):
        dev = None
        for a in (*args, *kwargs.values()):
            if isinstance(a, torch.Tensor):
                dev = a.device
                break
        if dev is None or dev.type != "mps":
            return orig(*args, **kwargs)
        cargs = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        ckwargs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
        return orig(*cargs, **ckwargs).to(dev)

    return wrapped


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    try:
        import comfy_kitchen  # noqa: F401  (ensures backends register)
        from comfy_kitchen.registry import registry
        import comfy_kitchen.backends.eager.quantization as qmod
    except Exception:
        return  # comfy_kitchen not installed; nothing to patch

    # separate from the gate above so a build without it still gets the fp8 fix
    try:
        import comfy_kitchen.float_utils as fumod
    except ImportError:
        fumod = None

    eager = registry._backends.get("eager")
    if eager is None:
        return

    orig_dequant = eager.dequantize_per_tensor_fp8
    orig_quant = eager.quantize_per_tensor_fp8

    def dequantize_per_tensor_fp8(x, scale, output_type=torch.bfloat16):
        if x.device.type == "mps" and x.dtype in FP8_DTYPES:
            return decode_fp8(x).to(output_type) * scale.to(output_type)
        return orig_dequant(x, scale, output_type)

    def quantize_per_tensor_fp8(x, scale, output_type=torch.float8_e4m3fn):
        if x.device.type == "mps" and output_type in FP8_DTYPES:
            lp_max = (
                qmod.F8_E4M3_MAX if output_type == torch.float8_e4m3fn else qmod.F8_E5M2_MAX
            )
            temp = torch.clamp(x * (1.0 / scale).to(x.dtype), -lp_max, lp_max)
            return temp.to("cpu").to(output_type).to(x.device)  # FP8 cast unsupported on MPS
        return orig_quant(x, scale, output_type)

    for mod in (eager, qmod):
        mod.dequantize_per_tensor_fp8 = dequantize_per_tensor_fp8
        mod.quantize_per_tensor_fp8 = quantize_per_tensor_fp8

    # getattr-guarded: a no-op on comfy_kitchen builds predating these formats
    nvfp4_mxfp8 = []
    for fname in ("dequantize_nvfp4", "dequantize_mxfp8"):
        orig_fn = getattr(eager, fname, None)
        if orig_fn is None:
            continue
        wrapped = _cpu_dequant_on_mps(orig_fn)
        for mod in (eager, qmod):
            setattr(mod, fname, wrapped)
        nvfp4_mxfp8.append(fname)

    # to_blocked pads via a strided fp8 copy MPS has no kernel for. The dtype guard
    # leaves MXFP8's uint8 E8M0 scales alone; patch every module resolving the name.
    orig_to_blocked = getattr(fumod, "to_blocked", None) if fumod is not None else None
    if orig_to_blocked is not None:
        def to_blocked(input_matrix, *args, **kwargs):
            if input_matrix.device.type == "mps" and input_matrix.dtype in FP8_DTYPES:
                return orig_to_blocked(input_matrix.cpu(), *args, **kwargs).to(input_matrix.device)
            return orig_to_blocked(input_matrix, *args, **kwargs)

        for mod in (fumod, qmod, comfy_kitchen):
            if getattr(mod, "to_blocked", None) is not None:
                mod.to_blocked = to_blocked
        nvfp4_mxfp8.append("to_blocked")

    _installed = True
    extra = f" (+ {', '.join(nvfp4_mxfp8)} via CPU)" if nvfp4_mxfp8 else ""
    print(f"{TAG} patched comfy_kitchen eager FP8 dequantize/quantize for MPS{extra}.")
