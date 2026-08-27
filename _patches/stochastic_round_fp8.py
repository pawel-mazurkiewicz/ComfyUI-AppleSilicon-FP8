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

Fix: wrap comfy.float.stochastic_rounding and keep the work on the GPU where
possible. Only the *last* step of the re-quant is unsupported on MPS — the
float→FP8 cast — and patch #8 (tensor_to_fp8) already routes that single op via
a LUT/CPU hop, so on path (a) the whole function runs on MPS and returns
byte-identical FP8.

Path (b) still needs the old treatment: it writes through
``output[i:].copy_(...)``, a strided FP8 copy patch #8 does not cover (it wraps
``.to``, not ``copy_``). So we try the native path once, and latch onto the CPU
round-trip for the session if it raises.

This matters because LoRA application re-quantises every weight it touches.
Sending each one to the CPU in full measured ~5.6x slower at per-weight
granularity (11.3 → 2.0 ms/weight, M5 Max), which is why loading an fp8 DiT with
a LoRA took 80–200 s while the same model without one loaded fast (issue #29).

The same module's `to_blocked` needs the same treatment (issue #8). It is the
swizzle at the end of the NVFP4 *quantize* direction, which an ordinary NVFP4
checkpoint load reaches with no LoRA involved:

    ops.MixedPrecisionOps.Linear.set_weight
      -> QuantizedTensor.requantize_from_float
      -> TensorCoreNVFP4Layout.quantize
      -> comfy.float.stochastic_round_quantize_nvfp4_by_block
      -> to_blocked(fp8 block-scales)

to_blocked pads the block-scale matrix out to a multiple of (128, 4) with
`padded[:rows, :cols] = input_matrix`. When the columns need padding — i.e.
in_features % 64 != 0 — that destination slice is non-contiguous, and MPS has no
strided fp8 copy kernel:

    RuntimeError: Undefined type Float8_e4m3fn

(fp8_mps_strided covers the *later* reshape-after-permute in the same function,
but not this __setitem__.) The rearrangement is pure data movement, so the CPU
round-trip is bit-exact, and the block-scale matrix is small. MXFP8 feeds uint8
E8M0 scales through the same function; MPS handles those natively, so the dtype
guard leaves them alone.
"""

import torch

from ._common import FP8_DTYPES

TAG = "[AppleSilicon-FP8/stochastic_round]"

_installed = False

# None = untried, True = the GPU path works here, False = this stack needs the CPU
# round-trip. Latched: a LoRA re-quantises every weight it touches, so probing per
# weight would put the exception cost straight back on the path #29 is about.
_native_ok = None


def _requant(original, value, dtype, seed=0):
    """Re-quantise `value` to an fp8 `dtype`, keeping the maths on the GPU if it can.

    Only ONE step of the re-quant has no MPS kernel -- the final float->fp8 cast.
    Patch #8 (tensor_to_fp8) already routes that single op via a LUT/CPU hop, so
    on a stack where it is installed the whole function runs on the GPU and
    returns byte-identical fp8. Sending the entire tensor to the CPU instead, as
    this patch did from S1 until #29, costs ~5.6x at per-weight granularity
    (11.3 -> 2.0 ms/weight on an M5 Max) and is why loading an fp8 DiT with a
    LoRA took minutes while the same model without one loaded fast.

    The CPU round-trip stays reachable because it is still needed: comfy's own
    fallback implementation writes through ``output[i:].copy_(...)``, a strided
    fp8 copy that patch #8 does not cover (it wraps ``.to``, not ``copy_``).
    """
    global _native_ok
    if value.device.type != "mps" or dtype not in FP8_DTYPES:
        return original(value, dtype, seed=seed)

    if _native_ok is not False:
        try:
            out = original(value, dtype, seed=seed)
            _native_ok = True
            return out
        except Exception as e:
            if _native_ok is None:
                print(f"{TAG} fp8 re-quant cannot run on MPS here ({e!r}); "
                      f"using the CPU round-trip for the rest of this session.")
            _native_ok = False

    # float->fp8 casts are fully supported on CPU; the fp8 result moves back as
    # plain storage, which MPS handles.
    return original(value.cpu(), dtype, seed=seed).to(value.device)


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
        return _requant(_original, value, dtype, seed)

    float_mod.stochastic_rounding = _mps_safe_stochastic_rounding

    extra = ""
    _orig_to_blocked = getattr(float_mod, "to_blocked", None)
    if _orig_to_blocked is not None:
        def _mps_safe_to_blocked(input_matrix, *args, **kwargs):
            if input_matrix.device.type == "mps" and input_matrix.dtype in FP8_DTYPES:
                return _orig_to_blocked(input_matrix.cpu(), *args, **kwargs).to(input_matrix.device)
            return _orig_to_blocked(input_matrix, *args, **kwargs)

        float_mod.to_blocked = _mps_safe_to_blocked
        extra = " + to_blocked"

    _installed = True
    print(f"{TAG} stochastic_rounding{extra} FP8 re-quant routed via CPU on MPS.")
