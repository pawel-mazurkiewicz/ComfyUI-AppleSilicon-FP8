"""Fix: FP8 stochastic rounding on MPS (LoRA re-quant of an fp8 base model).

Wraps comfy.float.stochastic_rounding and to_blocked: try the GPU once, then latch onto
a bit-exact CPU round-trip for the session if this stack can't run the op natively.
"""

import torch

from ._common import FP8_DTYPES

TAG = "[AppleSilicon-FP8/stochastic_round]"

_installed = False

# None = untried, True = GPU works here, False = this stack needs the CPU round-trip
_native_ok = None


def _is_transient(e):
    """Memory pressure rather than a missing kernel, so not worth latching off."""
    oom = getattr(torch, "OutOfMemoryError", None)
    if oom is not None and isinstance(e, oom):
        return True
    return "out of memory" in str(e).lower()


def _requant(original, value, dtype, seed=0):
    """Re-quantise `value` to an fp8 `dtype`, keeping the maths on the GPU if it can.

    Whether the GPU path works depends on which implementation comfy calls, which we
    cannot tell in advance, so try it once and latch the verdict for the session.
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
            # deliberately broad: the CPU fallback below is bit-exact, so re-raising
            # would only turn a logged slowdown into a failed model load
            if _is_transient(e):
                return original(value.cpu(), dtype, seed=seed).to(value.device)
            if _native_ok is None:
                print(f"{TAG} fp8 re-quant cannot run on MPS here ({e!r}); "
                      f"using the CPU round-trip for the rest of this session.")
            _native_ok = False

    # float->fp8 casts work on CPU, and the fp8 result moves back as plain storage
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
    print(f"{TAG} stochastic_rounding{extra} FP8 re-quant runs on the GPU where "
          f"the stack allows it, with a CPU round-trip as the fallback.")
