"""Capability probes plus the three-state env gate that decides what installs by default.

A perf patch is default-on only where the hardware/software can run it; an explicit env
var always wins over the probe. See `resolve` for the three states.
"""

import os
import platform
import threading

import torch

# anything outside these vocabularies is treated as "unset"
ON_TOKENS = ("1", "on", "true", "yes", "enable", "enabled")
OFF_TOKENS = ("0", "off", "false", "no", "disable", "disabled")


def is_mps():
    """True iff a usable MPS backend is present."""
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


_compile_shader = None


def has_compile_shader():
    """Tier A: MPS + `torch.mps.compile_shader` (the fp32 kernels need nothing more)."""
    global _compile_shader
    if _compile_shader is None:
        _compile_shader = bool(is_mps() and hasattr(torch.mps, "compile_shader"))
    return _compile_shader


_tensor_ops = None


def has_tensor_ops_matmul2d():
    """Tier A+: does na_gemm's matmul2d shader compile AND compute correctly?

    Only for shaders built through `torch.mps.compile_shader` (#19 conv im2col);
    the ObjC++ extensions gate on kernel_gate() instead.
    """
    global _tensor_ops
    if _tensor_ops is None:
        if not has_compile_shader():
            _tensor_ops = False
        else:
            try:
                from . import na_gemm
                _tensor_ops = bool(na_gemm.self_check_ok())
            except Exception:
                _tensor_ops = False
    return _tensor_ops


_ninja = None


def ninja_available():
    """True iff `ninja` is reachable, which torch's cpp_extension needs to build."""
    global _ninja
    if _ninja is None:
        import shutil
        ok = shutil.which("ninja") is not None
        if not ok:
            try:
                import importlib.util
                ok = importlib.util.find_spec("ninja") is not None
            except Exception:
                ok = False
        _ninja = bool(ok)
    return _ninja


_UNPROBED = object()
_chip_gen = _UNPROBED

# first generation with the Neural Accelerators matmul2d needs; M1-M4 compile the
# same shader and return garbage
_MATRIX_UNIT_GEN = 5


def _cpu_brand_string():
    """The chip's marketing name ('Apple M5 Max'), or None if it can't be read."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
    except Exception:
        return None
    return out.decode(errors="replace").strip() or None


def chip_generation():
    """Apple Silicon generation as an int ('Apple M4 Pro' -> 4), or None for "don't know"."""
    global _chip_gen
    if _chip_gen is _UNPROBED:
        import re
        brand = _cpu_brand_string()
        m = re.match(r"^Apple M(\d+)", brand) if brand else None
        _chip_gen = int(m.group(1)) if m else None
    return _chip_gen


def has_neural_accelerators():
    """Does this GPU have the M5-class matrix units the tensor-ops kernels need?

    Only a chip positively identified as pre-M5 answers False: a false positive costs
    one build that kernel_ready() then rejects, a false negative disables everything.
    """
    gen = chip_generation()
    return gen is None or gen >= _MATRIX_UNIT_GEN


def kernel_gate():
    """Default-on pre-filter for the ObjC++ tensor-ops extensions (#3, #17).

    Deliberately skips has_tensor_ops_matmul2d(): that probe reports the torch build's
    default MSL version, while these extensions compile through `newLibraryWithSource`
    at an explicit one. Says a build is worth attempting, not that a kernel works —
    that is kernel_ready()'s job.
    """
    return is_mps() and has_neural_accelerators() and ninja_available()


# name -> None (untried) | True (verified working) | False (verified broken)
_kernel_ready = {}

# reentrant: verify_fn is arbitrary caller code. Lock order is always this lock then
# _extbuild._BUILD_LOCK, never the reverse.
_kernel_lock = threading.RLock()


def kernel_ready(name, verify_fn):
    """Has THIS kernel been proven to work end to end on THIS machine?

    `verify_fn` must do the real thing: build, warmup, check numerics. Memoised per
    name including failures, so a broken kernel isn't rebuilt by every layer.
    """
    with _kernel_lock:
        cached = _kernel_ready.get(name)
        if cached is not None:
            return cached
        try:
            ok = bool(verify_fn())
        except Exception as e:
            print(f"[AppleSilicon-FP8] {name} kernel verification raised; "
                  f"disabling it for this session: {e!r}", flush=True)
            ok = False
        _kernel_ready[name] = ok
        return ok


def resolve(env_name, default_on, cap):
    """Three-state gate. Returns True iff the patch should install.

    `cap` is evaluated only when the env var is unset or unrecognised, so an explicit
    on/off never pays for a probe and forced-on wins over a failing probe.
    """
    v = os.environ.get(env_name)
    if v is not None:
        t = v.strip().lower()
        if t in OFF_TOKENS:
            return False
        if t in ON_TOKENS:
            return True
    return bool(default_on and cap())


def mark_kernel_failed(name):
    """Latch off a kernel that passed verification but then failed in use.

    A dispatch failure will repeat on every later layer, and falling back is always
    correct.
    """
    with _kernel_lock:
        _kernel_ready[name] = False


def reset_cache():
    """Test hook: forget memoised probe results so a test can re-probe."""
    global _compile_shader, _tensor_ops, _ninja, _chip_gen
    _compile_shader = _tensor_ops = _ninja = None
    _chip_gen = _UNPROBED
    # has_tensor_ops_matmul2d() reads through to na_gemm's own memo
    try:
        from . import na_gemm
        na_gemm.reset_cache()
    except Exception:
        pass
    with _kernel_lock:
        _kernel_ready.clear()


def summary():
    """One-line capability banner for the startup log.

    Every field is cheap; na_gemm's shader probe is left to its one consumer.
    """
    matrix = "unknown" if chip_generation() is None else (
        "yes" if has_neural_accelerators() else "no")
    return ", ".join((
        f"macOS={platform.mac_ver()[0] or '?'}",
        f"torch={torch.__version__}",
        f"mps={'yes' if is_mps() else 'no'}",
        f"compile_shader={'yes' if has_compile_shader() else 'no'}",
        f"chip={_cpu_brand_string() or 'unknown'}",
        f"matrix_units(M5+)={matrix}",
        f"ninja={'yes' if ninja_available() else 'no'}",
    ))
