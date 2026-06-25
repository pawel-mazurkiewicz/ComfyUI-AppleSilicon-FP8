"""DIAGNOSTIC (opt-in, ASFP8_TRACE_OPS=1): trace which low-level matmul op a model
actually dispatches to.

Mixed-precision checkpoints (e.g. a file named "..._fp8_scaled" whose metadata is
really `int8_tensorwise` native + fp8 emulated) make it ambiguous whether the
compute hits `torch._scaled_mm` (fp8), `torch._int_mm` (int8), or just decodes to
bf16 and runs a plain `F.linear`. Guessing the seam has burned us; this measures it.

Installs LAST (on top of whatever the other patches wrapped), so it sees every call
the model makes. For each of `torch._scaled_mm`, `torch._int_mm`, `F.linear` it logs
the operand dtypes/shapes on the first call and again at powers of two (so ~10 lines
per op over a full sampling run gives the magnitude), then delegates unchanged.

Pure observability — changes no numerics, builds nothing. Inert unless the flag is
set, so it is safe to leave wired in. Run one sampling pass with ASFP8_TRACE_OPS=1
and read which op dominates.
"""

import os

import torch

TAG = "[AppleSilicon-FP8/optrace]"

_installed = False
_counts = {}


def _pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def _desc(x):
    """Compact, never-throwing description of one operand."""
    try:
        name = type(x).__name__
        dt = getattr(x, "dtype", None)
        shp = list(getattr(x, "shape", ()) or ())
        dev = getattr(getattr(x, "device", None), "type", "?")
        return f"{name}<{dt}>{shp}@{dev}"
    except Exception:
        return type(x).__name__


def _log(op, operands):
    n = _counts.get(op, 0) + 1
    _counts[op] = n
    if n == 1 or _pow2(n):
        sig = ", ".join(_desc(o) for o in operands)
        print(f"{TAG} {op} call #{n}: {sig}")


def install():
    global _installed
    if _installed:
        return
    if os.environ.get("ASFP8_TRACE_OPS") != "1":
        return

    import torch.nn.functional as F

    if hasattr(torch, "_scaled_mm"):
        _smm = torch._scaled_mm

        def traced_scaled_mm(input, other, *args, **kwargs):
            _log("_scaled_mm", (input, other))
            return _smm(input, other, *args, **kwargs)

        torch._scaled_mm = traced_scaled_mm

    if hasattr(torch, "_int_mm"):
        _imm = torch._int_mm

        def traced_int_mm(a, b, *args, **kwargs):
            _log("_int_mm", (a, b))
            return _imm(a, b, *args, **kwargs)

        torch._int_mm = traced_int_mm

    _lin = F.linear

    def traced_linear(inp, weight, bias=None):
        _log("F.linear", (inp, weight))
        return _lin(inp, weight, bias)

    F.linear = traced_linear

    _installed = True
    print(f"{TAG} op tracer active: logging _scaled_mm / _int_mm / F.linear (first call + powers of two).")
