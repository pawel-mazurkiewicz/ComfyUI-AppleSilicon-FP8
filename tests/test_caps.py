"""Contract tests for the three-state capability gate (_patches/_caps.py).

These are pure/host-side: they exercise the env-resolution logic with a stubbed
capability predicate, so they run identically on CI (no MPS) and on an M5 box.
The gate is the mechanism every default-on perf patch now shares, so its contract
is worth pinning independently of any one kernel.
"""
import time

import pytest

from _patches import _caps


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Setup-only: monkeypatch auto-reverts, so no teardown/yield is needed.
    monkeypatch.delenv("ASFP8_CAPS_TEST", raising=False)


def _cap(value):
    """A zero-arg predicate that also records whether it was called."""
    calls = []

    def probe():
        calls.append(True)
        return value

    probe.calls = calls
    return probe


# --- unset: defers to default_on AND the capability probe -------------------
def test_unset_default_on_cap_true_installs():
    assert _caps.resolve("ASFP8_CAPS_TEST", default_on=True, cap=_cap(True)) is True


def test_unset_default_on_cap_false_skips():
    assert _caps.resolve("ASFP8_CAPS_TEST", default_on=True, cap=_cap(False)) is False


def test_unset_default_off_never_installs_even_if_capable():
    assert _caps.resolve("ASFP8_CAPS_TEST", default_on=False, cap=_cap(True)) is False


# --- explicit values win over the probe, and skip it entirely ---------------
@pytest.mark.parametrize("tok", _caps.ON_TOKENS)
def test_explicit_on_forces_install_without_probing(monkeypatch, tok):
    monkeypatch.setenv("ASFP8_CAPS_TEST", tok)
    probe = _cap(False)  # probe says "unsupported"; explicit ON must still win
    assert _caps.resolve("ASFP8_CAPS_TEST", default_on=True, cap=probe) is True
    assert probe.calls == [], "explicit ON must not pay for a capability probe"


@pytest.mark.parametrize("tok", _caps.OFF_TOKENS)
def test_explicit_off_forces_skip_without_probing(monkeypatch, tok):
    monkeypatch.setenv("ASFP8_CAPS_TEST", tok)
    probe = _cap(True)  # probe says "supported"; explicit OFF must still win
    assert _caps.resolve("ASFP8_CAPS_TEST", default_on=True, cap=probe) is False
    assert probe.calls == [], "explicit OFF must not pay for a capability probe"


def test_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv("ASFP8_CAPS_TEST", "  OFF  ")
    assert _caps.resolve("ASFP8_CAPS_TEST", default_on=True, cap=_cap(True)) is False


def test_unrecognised_value_falls_through_to_default(monkeypatch):
    monkeypatch.setenv("ASFP8_CAPS_TEST", "maybe")
    assert _caps.resolve("ASFP8_CAPS_TEST", default_on=True, cap=_cap(True)) is True
    assert _caps.resolve("ASFP8_CAPS_TEST", default_on=True, cap=_cap(False)) is False


def test_summary_is_a_string():
    s = _caps.summary()
    assert isinstance(s, str)
    assert "mps=" in s
    assert "ninja=" in s


# --- per-kernel verification (issue #14) -----------------------------------


def test_kernel_ready_runs_verify_once_and_memoises_success():
    _caps.reset_cache()
    calls = []

    def verify():
        calls.append(True)
        return True

    assert _caps.kernel_ready("probe-ok", verify) is True
    assert _caps.kernel_ready("probe-ok", verify) is True
    assert len(calls) == 1, "verify_fn must not re-run once the answer is known"


def test_kernel_ready_memoises_failure_too():
    """The point of the primitive: a known-bad kernel must not rebuild per call.

    Re-running verify on every eligible layer is what made issue #13 cost 1.46x
    instead of merely disabling int8.
    """
    _caps.reset_cache()
    calls = []

    def verify():
        calls.append(True)
        return False

    assert _caps.kernel_ready("probe-bad", verify) is False
    assert _caps.kernel_ready("probe-bad", verify) is False
    assert len(calls) == 1, "a failed kernel was re-verified"


def test_kernel_ready_treats_a_raising_verify_as_failure():
    _caps.reset_cache()

    def verify():
        raise RuntimeError("toolchain rejected the shader")

    assert _caps.kernel_ready("probe-raise", verify) is False


def test_kernel_ready_is_per_name():
    _caps.reset_cache()
    assert _caps.kernel_ready("a", lambda: True) is True
    assert _caps.kernel_ready("b", lambda: False) is False
    assert _caps.kernel_ready("a", lambda: False) is True, "names must not share state"


def test_reset_cache_clears_the_kernel_results():
    _caps.reset_cache()
    calls = []

    def verify():
        calls.append(True)
        return True

    _caps.kernel_ready("probe-reset", verify)
    _caps.reset_cache()
    _caps.kernel_ready("probe-reset", verify)
    assert len(calls) == 2, "reset_cache() must force a re-probe"


def test_summary_banner_reports_the_chip_and_matrix_units():
    """The banner is what users paste into bug reports, so it has to name the
    thing the kernels actually depend on. `tensor_ops(M5/Metal4)=` claimed a GPU
    generation it never measured -- it read yes on the M4 Pro of #25 and no on the
    M5 Max of #27."""
    s = _caps.summary()
    for token in ("mps=", "chip=", "matrix_units(M5+)=", "ninja="):
        assert token in s, f"{token!r} missing from banner: {s}"
    assert "tensor_ops(M5/Metal4)=" not in s


def test_summary_says_unknown_when_the_chip_cannot_be_identified(monkeypatch):
    """Permissive gate, honest banner: an unidentified chip still gets a build
    attempt, but the banner must not claim matrix units we never confirmed."""
    _caps.reset_cache()
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: None)
    s = _caps.summary()
    assert "matrix_units(M5+)=unknown" in s


def test_kernel_ready_verifies_once_under_concurrency():
    """Two layers hitting an unverified kernel at the same time must not each
    start their own extension build.

    The whole sequence -- lookup, verify, store -- has to be inside the lock; a
    bare dict check leaves both callers seeing None and both building.
    """
    import threading

    _caps.reset_cache()
    calls = []
    both_ready = threading.Barrier(2, timeout=30)

    def verify():
        calls.append(1)
        time.sleep(0.05)          # widen the window a racing caller slips through
        return True

    results = []

    def worker():
        both_ready.wait()
        results.append(_caps.kernel_ready("concurrent", verify))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not any(t.is_alive() for t in threads)
    assert results == [True, True]
    assert len(calls) == 1, f"verify_fn ran {len(calls)}x for concurrent callers"


def test_mark_kernel_failed_disables_without_reverifying():
    _caps.reset_cache()
    calls = []

    def verify():
        calls.append(1)
        return True

    assert _caps.kernel_ready("latch", verify) is True
    _caps.mark_kernel_failed("latch")
    assert _caps.kernel_ready("latch", verify) is False
    assert len(calls) == 1, "a latched-off kernel was re-verified"


# --- chip identification and the matrix-unit gate (issues #25, #27) ---------
#
# Both issues are the same defect: has_tensor_ops_matmul2d() compiles na_gemm's
# bf16 shader through torch.mps.compile_shader, which cannot request an MSL
# language version -- so its answer tracks the torch build's default MSL, not the
# GPU. #25 got tensor_ops=yes on an M4 Pro (kernel builds, every element garbage);
# #27 got tensor_ops=no on an M5 Max (kernel is bit-exact and 3.17x faster). The
# ObjC++ kernels want one thing the probe never measured: M5-class matrix units.


@pytest.mark.parametrize("brand,gen", [
    ("Apple M1", 1),
    ("Apple M2 Pro", 2),
    ("Apple M4 Pro", 4),
    ("Apple M5 Max", 5),
    ("Apple M10 Ultra", 10),
])
def test_chip_generation_parses_the_brand_string(monkeypatch, brand, gen):
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: brand)
    assert _caps.chip_generation() == gen


@pytest.mark.parametrize("brand", [
    "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz",
    "Apple silicon",
    "",
    None,
])
def test_chip_generation_is_none_when_the_chip_is_unrecognisable(monkeypatch, brand):
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: brand)
    assert _caps.chip_generation() is None


def test_chip_generation_is_memoised(monkeypatch):
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    calls = []

    def brand():
        calls.append(1)
        return "Apple M5 Max"

    monkeypatch.setattr(_caps, "_cpu_brand_string", brand)
    assert _caps.chip_generation() == 5
    assert _caps.chip_generation() == 5
    assert len(calls) == 1, "the sysctl probe must run once per session"


def test_m4_reports_no_neural_accelerators(monkeypatch):
    """#25: the tensor_ops kernels compile on M4 and return garbage."""
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: "Apple M4 Pro")
    assert _caps.has_neural_accelerators() is False


def test_m5_reports_neural_accelerators(monkeypatch):
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: "Apple M5 Max")
    assert _caps.has_neural_accelerators() is True


def test_unidentified_chip_stays_permissive(monkeypatch):
    """Never false-negative on hardware we cannot name -- that is #27's failure
    mode. kernel_ready()'s build + numeric self-check is the real authority; the
    chip check only short-circuits hardware we positively know cannot work."""
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: None)
    assert _caps.has_neural_accelerators() is True


# --- kernel_gate: the pre-filter the ObjC++ extensions actually want --------


def test_kernel_gate_requires_neural_accelerators(monkeypatch):
    monkeypatch.setattr(_caps, "is_mps", lambda: True)
    monkeypatch.setattr(_caps, "ninja_available", lambda: True)
    monkeypatch.setattr(_caps, "has_neural_accelerators", lambda: False)
    assert _caps.kernel_gate() is False


def test_kernel_gate_requires_ninja(monkeypatch):
    monkeypatch.setattr(_caps, "is_mps", lambda: True)
    monkeypatch.setattr(_caps, "has_neural_accelerators", lambda: True)
    monkeypatch.setattr(_caps, "ninja_available", lambda: False)
    assert _caps.kernel_gate() is False


def test_kernel_gate_requires_mps(monkeypatch):
    monkeypatch.setattr(_caps, "has_neural_accelerators", lambda: True)
    monkeypatch.setattr(_caps, "ninja_available", lambda: True)
    monkeypatch.setattr(_caps, "is_mps", lambda: False)
    assert _caps.kernel_gate() is False


def test_kernel_gate_passes_on_m5_with_a_toolchain(monkeypatch):
    monkeypatch.setattr(_caps, "is_mps", lambda: True)
    monkeypatch.setattr(_caps, "has_neural_accelerators", lambda: True)
    monkeypatch.setattr(_caps, "ninja_available", lambda: True)
    assert _caps.kernel_gate() is True


def test_kernel_gate_ignores_the_na_gemm_compile_probe(monkeypatch):
    """#27 in one assertion: the M5 Max where compile_shader could not build
    na_gemm's bf16 shader is the same M5 Max where int8_gemm.mm is bit-exact.
    The ObjC++ kernels compile through newLibraryWithSource at an explicit MSL
    version, so what compile_shader can manage says nothing about them."""
    monkeypatch.setattr(_caps, "is_mps", lambda: True)
    monkeypatch.setattr(_caps, "has_neural_accelerators", lambda: True)
    monkeypatch.setattr(_caps, "ninja_available", lambda: True)

    probed = []

    def probe():
        probed.append(1)
        return False

    monkeypatch.setattr(_caps, "has_tensor_ops_matmul2d", probe)
    assert _caps.kernel_gate() is True
    assert probed == [], "kernel_gate() must not consult the compile_shader probe"


# --- the compile_shader probe must verify numerics, not just compilation ----


def test_tensor_ops_probe_requires_a_passing_numeric_self_check(monkeypatch):
    """na_gemm.available() only proves the shader compiled. #25 is a machine where
    a tensor_ops shader compiles and computes garbage, so conv_im2col -- the one
    consumer still gated on this probe -- needs the numeric check, not the build."""
    import sys
    import types

    monkeypatch.setattr(_caps, "_tensor_ops", None)
    monkeypatch.setattr(_caps, "has_compile_shader", lambda: True)

    import _patches

    fake = types.ModuleType("_patches.na_gemm")
    fake.available = lambda: True
    fake.self_check_ok = lambda: False
    # `from . import na_gemm` resolves through the parent package attribute once
    # the real module has been imported, so both bindings have to be replaced.
    monkeypatch.setitem(sys.modules, "_patches.na_gemm", fake)
    monkeypatch.setattr(_patches, "na_gemm", fake, raising=False)

    assert _caps.has_tensor_ops_matmul2d() is False


def test_no_patch_seeds_the_global_rng():
    """`torch.manual_seed()` inside a library is never right: it reseeds every
    device for the whole host process. The capability probes and kernel
    self-checks all need deterministic operands, which is what a local
    torch.Generator is for -- scaled_mm_fp8's fp8 self-check runs lazily on the
    *first fp8 matmul*, so seeding there lands in the middle of a render."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "_patches"
    offenders = [
        f"{p.relative_to(root.parent)}:{i}"
        for p in sorted(root.rglob("*.py"))
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if "torch.manual_seed(" in line and not line.lstrip().startswith("#")
    ]
    assert offenders == [], f"global RNG seeded by: {offenders}"
