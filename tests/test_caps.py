"""Contract tests for the three-state capability gate (_patches/_caps.py).

These are pure/host-side: they exercise the env-resolution logic with a stubbed
capability predicate, so they run identically on CI (no MPS) and on an M5 box.
The gate is the mechanism every default-on perf patch now shares, so its contract
is worth pinning independently of any one kernel.
"""
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


# --- tier B implies both the tensor-ops probe AND ninja ---------------------
def test_tier_b_requires_both_probe_and_ninja(monkeypatch):
    monkeypatch.setattr(_caps, "has_tensor_ops_matmul2d", lambda: True)
    monkeypatch.setattr(_caps, "ninja_available", lambda: False)
    assert _caps.tier_b_ready() is False
    monkeypatch.setattr(_caps, "ninja_available", lambda: True)
    assert _caps.tier_b_ready() is True


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


def test_summary_banner_substrings_are_unchanged():
    """Other tests (and users' bug reports) match on these exact substrings."""
    s = _caps.summary()
    for token in ("mps=", "tensor_ops(M5/Metal4)=", "ninja="):
        assert token in s, f"{token!r} missing from banner: {s}"
