"""Hardware/software capability probes + a three-state env gate.

The one place that decides, per patch, whether an acceleration installs by DEFAULT.
The promise this module encodes: a performance patch is on by default ONLY on
hardware/software that can actually run it; everywhere else it stays inert. An
explicit env var always wins over the probe (power-user override), so nothing here
can lock a user out or force a broken kernel on.

Three states per gate (see `resolve`):
  <VAR> unset          -> default_on AND the capability probe passes
  <VAR> in OFF tokens  -> OFF   (force disable; probe skipped)
  <VAR> in ON tokens   -> ON    (force enable; probe skipped — the patch's own
                                  loader still no-ops if the kernel can't build)

Capability tiers:
  A  `torch.mps.compile_shader` present            -> fp32 compile_shader kernels
     (has_compile_shader)                              (#18 fused-norm, #21 RoPE)
  A+ na_gemm's tensor-ops shader computes right    -> compile_shader matmul kernels
     (has_tensor_ops_matmul2d)                         (#19 conv im2col)
  B  M5-class matrix units + `ninja`               -> ObjC++ cpp_extension kernels
     (kernel_gate)                                     (#3 fp8_ext, #17 int8_ext, int4)

A+ and B are separate ladders, not steps of one. A+ asks what
`torch.mps.compile_shader` can build; B's kernels compile through
`newLibraryWithSource` at an explicit MSL version and never touch compile_shader.
Conflating them is issues #25 and #27 — the same probe answered yes on an M4 Pro
that computes garbage and no on an M5 Max that is bit-exact.
"""

import os
import platform
import threading

import torch

# Token vocabularies for the three-state gate. Anything else is treated as "unset".
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
    """Does na_gemm's `mpp::tensor_ops::matmul2d` shader compile AND compute
    correctly on this stack? Delegates to na_gemm (memoised there) so we don't
    carry a second, drift-prone copy of the MPP shader source.

    Scope: this answers for shaders built through `torch.mps.compile_shader`,
    which is conv_im2col (#19) and nothing else. It is NOT the gate for the
    ObjC++ extensions -- see kernel_gate() for why compile_shader's verdict says
    nothing about them.

    The numeric self-check, not just the build, is what's consulted: issue #25 is
    a machine where a tensor_ops shader compiles cleanly and computes garbage, so
    "it compiled" was never enough to route real work through it."""
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
    """True iff the `ninja` build tool is reachable — required by torch's
    cpp_extension to build the ObjC++ Metal kernels (#3/#17/#20).

    Says a build is possible, not that any particular one succeeds."""
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

# First Apple Silicon generation whose GPU cores carry Neural Accelerators, the
# execution unit `mpp::tensor_ops::matmul2d` needs to produce correct results. On
# M1-M4 the same shader compiles and dispatches, and returns garbage (#25).
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
    """Apple Silicon generation as an int ('Apple M4 Pro' -> 4).

    None when this isn't an Apple Silicon Mac or the name doesn't parse -- an
    answer callers must read as "don't know", never as "no"."""
    global _chip_gen
    if _chip_gen is _UNPROBED:
        import re
        brand = _cpu_brand_string()
        m = re.match(r"^Apple M(\d+)", brand) if brand else None
        _chip_gen = int(m.group(1)) if m else None
    return _chip_gen


def has_neural_accelerators():
    """Does this GPU have the M5-class matrix units the tensor-ops kernels need?

    An unidentified chip answers True. The two failure modes are not symmetric: a
    false negative silently disables every ObjC++ kernel on hardware that runs
    them bit-exactly (#27), while a false positive costs one build that
    kernel_ready()'s numeric self-check then rejects (#25). So only a chip we
    positively identify as pre-M5 short-circuits."""
    gen = chip_generation()
    return gen is None or gen >= _MATRIX_UNIT_GEN


def kernel_gate():
    """Default-on pre-filter for the ObjC++ tensor-ops extensions (#3 fp8_ext,
    #17 int8_ext, int4): M5-class matrix units on MPS, plus a build toolchain.

    Deliberately does NOT consult has_tensor_ops_matmul2d(). That probe compiles
    a bf16 shader through `torch.mps.compile_shader`, which cannot request an MSL
    language version, so what it reports is the torch build's default MSL rather
    than anything about the GPU -- it read yes on the M4 Pro of #25, where the
    int8 kernel returns garbage, and no on the M5 Max of #27, where the same
    kernel is bit-exact and 3.17x faster. These extensions compile through
    `newLibraryWithSource` at an explicit version, so compile_shader's verdict was
    never theirs to give.

    A pre-filter only: it says a build is worth attempting, never that a given
    kernel works. That stays kernel_ready()'s job."""
    return is_mps() and has_neural_accelerators() and ninja_available()


# name -> None (untried) | True (verified working) | False (verified broken)
_kernel_ready = {}

# Held across the whole check-run-store sequence, so concurrent callers verify a
# kernel once rather than each starting their own extension build. Reentrant
# because a verify_fn is arbitrary caller code; the lock order is always this
# lock then _extbuild._BUILD_LOCK, never the reverse, so the two cannot deadlock.
_kernel_lock = threading.RLock()


def kernel_ready(name, verify_fn):
    """Has THIS kernel been proven to work end to end on THIS machine?

    kernel_gate() answers a different, weaker question (see its docstring), so
    a kernel that passes it can still fail to build. `verify_fn` must therefore
    do the real thing: build the extension, run its warmup(), and check its
    numerics. Only then is the kernel enabled.

    Memoised per name, failures included. That matters as much as the check
    itself: without a remembered failure every eligible layer retries the build,
    which is what made issue #13 cost 1.46x rather than merely disabling int8.
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

    `cap` is a zero-arg predicate evaluated ONLY when the env var is unset (or set
    to something unrecognised) — so an explicit on/off never pays for a probe and a
    forced-on value is honoured even where the probe would say no."""
    v = os.environ.get(env_name)
    if v is not None:
        t = v.strip().lower()
        if t in OFF_TOKENS:
            return False
        if t in ON_TOKENS:
            return True
        # Unrecognised value -> fall through to default + capability probe.
    return bool(default_on and cap())


def mark_kernel_failed(name):
    """Disable a kernel that passed verification but then failed in use.

    warmup() and the numeric self-check both passed before it was enabled, so a
    failure at dispatch is unexpected rather than routine, and it will almost
    certainly repeat on the next layer. Latching it costs some speed on the
    remainder of the render and buys back a per-layer exception plus its log
    line -- the 822 fallback lines in issue #13. Falling back is always correct.
    """
    with _kernel_lock:
        _kernel_ready[name] = False


def reset_cache():
    """Test hook: forget memoised probe results so a test can re-probe."""
    global _compile_shader, _tensor_ops, _ninja, _chip_gen
    _compile_shader = _tensor_ops = _ninja = None
    _chip_gen = _UNPROBED
    with _kernel_lock:
        _kernel_ready.clear()


def summary():
    """One-line capability banner for the startup log.

    Reports the chip rather than the old `tensor_ops(M5/Metal4)=` field, which
    named a GPU generation it never measured (yes on the M4 Pro of #25, no on the
    M5 Max of #27) and cost a shader compile on the import thread to get wrong.
    Every field here is cheap; na_gemm's probe is left to its one consumer to
    evaluate lazily."""
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
