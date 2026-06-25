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

Newer comfy_kitchen builds also ship microscaling layouts (NVFP4, MXFP8) whose
dequant unswizzles fp8 block-scales with a reshape-after-transpose; MPS can't make a
non-contiguous fp8 tensor contiguous ("Undefined type Float8_e4m3fn"), so those
dequants (`dequantize_nvfp4` / `dequantize_mxfp8`) are rerouted through the CPU and
the float result moved back. This is the path an NVFP4-quantized text encoder
(e.g. LTX's Gemma3) hits at encode time.

The registry resolves implementations via getattr() on the eager backend module
at call time, so overwriting the attributes there is picked up by every dispatch.
"""

import sys

import torch

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/comfy_kitchen]"


def _cpu_dequant_on_mps(orig):
    """Run a comfy_kitchen eager dequant on CPU when its inputs are on MPS, then move
    the (float) result back to the device.

    The microscaling layouts (NVFP4, MXFP8) unswizzle their fp8/e8m0 block-scales with
    `from_blocked`, which does a reshape-after-transpose. MPS cannot make a
    non-contiguous fp8 tensor contiguous (raises "Undefined type Float8_e4m3fn"), and
    the follow-up `block_scales.to(float)` is another unsupported fp8 cast. CPU has no
    such limit, the scales are tiny, and the returned dtype is float/bf16 (MPS-safe to
    move back), so doing the whole dequant off-device is correct and cheap.
    """
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

    # NVFP4 / MXFP8 microscaling dequant (newer comfy_kitchen): the block-scale
    # unswizzle does fp8 reshape-after-transpose that MPS can't execute. Run those
    # dequants on CPU and return the float result to the device. getattr-guarded so
    # this is a no-op on comfy_kitchen builds that predate these formats.
    nvfp4_mxfp8 = []
    for fname in ("dequantize_nvfp4", "dequantize_mxfp8"):
        orig_fn = getattr(eager, fname, None)
        if orig_fn is None:
            continue
        wrapped = _cpu_dequant_on_mps(orig_fn)
        for mod in (eager, qmod):
            setattr(mod, fname, wrapped)
        nvfp4_mxfp8.append(fname)

    extra = f" (+ {', '.join(nvfp4_mxfp8)} via CPU)" if nvfp4_mxfp8 else ""
    print(f"{TAG} patched comfy_kitchen eager FP8 dequantize/quantize for MPS{extra}.")
