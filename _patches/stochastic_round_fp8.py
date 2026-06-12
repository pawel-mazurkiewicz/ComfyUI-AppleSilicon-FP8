"""Fix: FP8 stochastic rounding on MPS (LoRA + FP8 base model).

When a LoRA is applied to an FP8-quantised base model, ComfyUI:
  1. Casts the stored FP8 weight to float32 on MPS (fine).
  2. Applies the LoRA delta in float32 (fine).
  3. Calls comfy.float.stochastic_rounding(result, float8_e4m3fn, seed)
     to re-quantise the patched weight back to FP8 for storage.

Step 3 crashes on MPS via two possible sub-paths:

  a) comfy_kitchen path  (when _CK_STOCHASTIC_ROUNDING_AVAILABLE):
       ck.stochastic_rounding_fp8(mps_tensor, rng, fp8_dtype)
     The eager backend ultimately does a float→FP8 cast on-device, which
     MPS does not support.

  b) Fallback path:
       output = torch.empty_like(value, dtype=fp8_dtype)   # FP8 on MPS — OK (storage)
       output[i:].copy_(manual_stochastic_round_to_float8(value[i:], ...))
     manual_stochastic_round_to_float8 returns a float16 MPS tensor;
     copy_ then has to convert float16→FP8 on-device — also unsupported.

Fix: wrap comfy.float.stochastic_rounding so that, when the input is on
MPS and the target dtype is FP8, the entire computation is moved to CPU
(where FP8 casts are fully supported). The resulting FP8 CPU tensor is
then moved back to MPS — storage works fine.
"""

import torch

from ._common import FP8_DTYPES

TAG = "[AppleSilicon-FP8/stochastic_round]"

_installed = False


def install():
    global _installed
    if _installed:
        return

    import sys
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    try:
        import comfy.float as float_mod
    except ImportError:
        return

    if not hasattr(float_mod, "stochastic_rounding"):
        return

    _original = float_mod.stochastic_rounding

    def _mps_safe_stochastic_rounding(value, dtype, seed=0):
        if value.device.type == "mps" and dtype in FP8_DTYPES:
            # Perform the full rounding on CPU where float→FP8 casts are
            # supported, then move the FP8 result back to MPS for storage.
            cpu_result = _original(value.cpu(), dtype, seed=seed)
            return cpu_result.to(value.device)
        return _original(value, dtype, seed=seed)

    float_mod.stochastic_rounding = _mps_safe_stochastic_rounding
    _installed = True
    print(f"{TAG} stochastic_rounding FP8 re-quant routed via CPU on MPS.")
