"""Fix: comfy_kitchen FP8 quantization on MPS (e.g. Ideogram 4).

Models quantized with ComfyUI's `comfy_kitchen` use its "eager" backend on
non-CUDA machines. That backend dequantizes/quantizes FP8 with plain casts:

    comfy_kitchen/backends/eager/quantization.py
        dequantize_per_tensor_fp8:  x.to(output_type) * scale.to(output_type)
        quantize_per_tensor_fp8:    temp.to(output_type)

On MPS those casts raise:

    TypeError: Trying to convert Float8_e4m3fn to the MPS backend but it does not
               have support for that dtype.

We replace the two eager functions with MPS-safe equivalents:
  * dequantize uses the LUT+gather decode (bit-identical to the original formula
    for both FP8 formats and float16/bfloat16/float32 outputs),
  * quantize does the unsupported float->FP8 final cast on CPU (rarely hit at
    inference; weights are already FP8 — this is a correctness safety net).

The registry resolves implementations via getattr() on the eager backend module
at call time, so overwriting the attributes there is picked up by every dispatch.
"""

import sys

import torch

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/comfy_kitchen]"


def install():
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

    print(f"{TAG} patched comfy_kitchen eager FP8 dequantize/quantize for MPS.")
