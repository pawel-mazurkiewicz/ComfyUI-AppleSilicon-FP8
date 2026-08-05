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
  B  Metal-4 tensor-ops `matmul2d` compiles         -> M5 / Metal-4.1 matmul kernels
     (has_tensor_ops_matmul2d, proven via na_gemm)     (#19 conv im2col)
     ...plus `ninja` for the ObjC++ cpp_extensions  -> #3 fp8_ext, #17 int8_ext,
     (tier_b_ready)                                     #20 fp8-native Linear
"""

import os
import platform

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
    """Tier B runtime probe: does `mpp::tensor_ops::matmul2d` compile on this
    OS/SDK/PyTorch/GPU? True only on M5-class / Metal-4.1 stacks. Delegates to
    na_gemm's proven-correct kernel (memoised there) so we don't carry a second,
    drift-prone copy of the MPP shader source.

    COARSE PRE-FILTER ONLY. This compiles na_gemm's bf16 shader, which is not the
    shader any of our int8/int4/fp8 kernels use — different operand types,
    different simdgroup counts. A pass proves the machine has Metal-4 tensor ops
    in general, NOT that a given kernel builds. Issue #13 was precisely that gap.
    Use kernel_ready() to find out whether a specific kernel actually works."""
    global _tensor_ops
    if _tensor_ops is None:
        if not has_compile_shader():
            _tensor_ops = False
        else:
            try:
                from . import na_gemm
                _tensor_ops = bool(na_gemm.available())
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


def tier_b_ready():
    """Metal-4 tensor ops present AND a build toolchain for the ObjC++ extensions.

    A cheap pre-filter for "is it worth attempting a build at all", nothing more.
    It green-lights every Tier-B kernel off one bf16 probe none of them use, so
    it can and does approve a kernel that will not compile (#13, #14). Gate the
    kernel itself on kernel_ready()."""
    return has_tensor_ops_matmul2d() and ninja_available()


# name -> None (untried) | True (verified working) | False (verified broken)
_kernel_ready = {}


def kernel_ready(name, verify_fn):
    """Has THIS kernel been proven to work end to end on THIS machine?

    tier_b_ready() answers a different, weaker question (see its docstring), so
    a kernel that passes it can still fail to build. `verify_fn` must therefore
    do the real thing: build the extension, run its warmup(), and check its
    numerics. Only then is the kernel enabled.

    Memoised per name, failures included. That matters as much as the check
    itself: without a remembered failure every eligible layer retries the build,
    which is what made issue #13 cost 1.46x rather than merely disabling int8.
    """
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


def reset_cache():
    """Test hook: forget memoised probe results so a test can re-probe."""
    global _compile_shader, _tensor_ops, _ninja
    _compile_shader = _tensor_ops = _ninja = None
    _kernel_ready.clear()


def summary():
    """One-line capability banner for the startup log."""
    return ", ".join((
        f"macOS={platform.mac_ver()[0] or '?'}",
        f"torch={torch.__version__}",
        f"mps={'yes' if is_mps() else 'no'}",
        f"compile_shader={'yes' if has_compile_shader() else 'no'}",
        f"tensor_ops(M5/Metal4)={'yes' if has_tensor_ops_matmul2d() else 'no'}",
        f"ninja={'yes' if ninja_available() else 'no'}",
    ))
