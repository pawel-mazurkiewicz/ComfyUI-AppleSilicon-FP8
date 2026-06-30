# tests/test_mps_profile_seams.py
"""Unit tests for the G1 additions to mps_profile.py.

Covers:
  - F.silu / F.gelu / F.glu accumulate into 'activation' bucket on MPS tensors
  - CPU tensors pass through unwrapped (no stats recorded — _is_mps guard)
  - install() with ASFP8_PROFILE=1 marks F.silu/F.gelu/F.glu as _asfp8_timed
  - _try_wrap_rope() finds and wraps module-level RoPE functions by canonical name
  - _try_wrap_rope() wraps _ideogram4_apply_rope_lowp (KJNodes Ideogram4 path)
  - _try_wrap_rope() patches alias occurrences by object identity (Pass 2)
  - Functions whose id() is in _rope_wrapped_ids are not re-wrapped
  - Already-wrapped functions (_asfp8_timed=True) are not double-wrapped
  - Callables without __code__ (builtins, partials) are skipped safely
  - *args-only functions (co_argcount == 0) with __code__ ARE wrapped
  - No-target scan: _rope_wrapped_ids stays empty when no rope names found
  - install() is a no-op when ASFP8_PROFILE != 1

All tests are MPS-conditional where required; lazy-scanner tests are CPU-safe.
"""
import sys
import time
import types
import functools

import pytest
import torch
import torch.nn.functional as F

from _patches import mps_profile

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _reset(monkeypatch):
    """Bring mps_profile back to a clean pre-install state for isolated tests.
    Suppresses the periodic dump by setting _t_last_dump far in the future."""
    monkeypatch.setattr(mps_profile, "_stats", {})
    monkeypatch.setattr(mps_profile, "_rope_wrapped_ids", set())
    monkeypatch.setattr(mps_profile, "_gguf_wrapped", False)
    monkeypatch.setattr(mps_profile, "_t_start", time.perf_counter())
    monkeypatch.setattr(mps_profile, "_t_last_dump", time.perf_counter() + 9999.0)


# ---------------------------------------------------------------------------
# 1. Activation seam — numerics unchanged after wrapping
# ---------------------------------------------------------------------------
@requires_mps
def test_activation_silu_value_unchanged(monkeypatch):
    _reset(monkeypatch)
    x = torch.randn(64, 128, device="mps")
    ref = F.silu(x.clone())
    wrapped = mps_profile._timed("activation", F.silu)
    out = wrapped(x)
    torch.mps.synchronize()
    assert torch.allclose(out, ref, atol=1e-5), "silu values changed after profiling wrap"


@requires_mps
def test_activation_gelu_value_unchanged(monkeypatch):
    _reset(monkeypatch)
    x = torch.randn(64, 128, device="mps")
    ref = F.gelu(x.clone())
    wrapped = mps_profile._timed("activation", F.gelu)
    out = wrapped(x)
    torch.mps.synchronize()
    assert torch.allclose(out, ref, atol=1e-5), "gelu values changed after profiling wrap"


@requires_mps
def test_activation_glu_value_unchanged(monkeypatch):
    _reset(monkeypatch)
    # F.glu requires last dim to be even; default dim=-1
    x = torch.randn(64, 128, device="mps")
    ref = F.glu(x.clone())
    wrapped = mps_profile._timed("activation", F.glu)
    out = wrapped(x)
    torch.mps.synchronize()
    assert torch.allclose(out, ref, atol=1e-5), "glu values changed after profiling wrap"


# ---------------------------------------------------------------------------
# 2. Activation seam — stats accumulate on MPS, silent on CPU
# ---------------------------------------------------------------------------
@requires_mps
def test_activation_stats_accumulate_on_mps(monkeypatch):
    _reset(monkeypatch)
    x = torch.randn(32, 64, device="mps")
    ws = mps_profile._timed("activation", F.silu)
    wg = mps_profile._timed("activation", F.gelu)
    ws(x)
    wg(x)
    ws(x)
    torch.mps.synchronize()
    assert "activation" in mps_profile._stats, "activation bucket not created"
    calls, total = mps_profile._stats["activation"]
    assert calls == 3, f"expected 3 activation calls, got {calls}"
    assert total > 0.0, "total GPU time should be > 0"


def test_activation_cpu_tensor_no_stats(monkeypatch):
    """Off-MPS tensors must pass through unchanged with no stats recorded."""
    _reset(monkeypatch)
    x = torch.randn(16, 32)  # CPU — _is_mps returns False
    wrapped = mps_profile._timed("activation", F.silu)
    out = wrapped(x)
    assert out.shape == x.shape, "shape must be preserved on CPU passthrough"
    assert "activation" not in mps_profile._stats, "CPU tensor must not record stats"


# ---------------------------------------------------------------------------
# 3. install() seam — F.silu/F.gelu/F.glu are marked _asfp8_timed after install
# ---------------------------------------------------------------------------
@requires_mps
def test_install_wraps_activation_globals(monkeypatch):
    """install() with ASFP8_PROFILE=1 must mark F.silu/F.gelu/F.glu as _asfp8_timed.

    This test exercises Change 5 (install seam) independently of the _timed tests
    above. It would fail if the F.* wrapping lines were accidentally omitted from
    install() even though _timed itself works correctly.
    """
    monkeypatch.setenv("ASFP8_PROFILE", "1")
    monkeypatch.setattr(mps_profile, "_installed", False)

    # install() globally rebinds the full set of profiled seams (matmul, linear,
    # sdpa, conv2d/3d, layer_norm, rms_norm, bmm, silu/gelu/glu). Snapshot and
    # restore *all* of them — restoring only the activations leaks Python timing
    # wrappers onto torch.matmul/F.linear into later test files, which then makes
    # torch.compile graph-break and fall back to the CPU C++ inductor backend.
    _saved_torch = {name: getattr(torch, name) for name in ("matmul", "bmm")}
    _f_names = [
        "scaled_dot_product_attention", "linear", "conv2d", "conv3d",
        "layer_norm", "silu", "gelu", "glu",
    ]
    if hasattr(F, "rms_norm"):
        _f_names.append("rms_norm")
    _saved_F = {name: getattr(F, name) for name in _f_names}
    try:
        mps_profile.install()
        assert getattr(F.silu, "_asfp8_timed", False), "F.silu not wrapped by install()"
        assert getattr(F.gelu, "_asfp8_timed", False), "F.gelu not wrapped by install()"
        assert getattr(F.glu, "_asfp8_timed", False), "F.glu not wrapped by install()"
    finally:
        for name, fn in _saved_torch.items():
            setattr(torch, name, fn)
        for name, fn in _saved_F.items():
            setattr(F, name, fn)
        monkeypatch.setattr(mps_profile, "_installed", False)


# ---------------------------------------------------------------------------
# 4. Lazy RoPE scanner — canonical names
# ---------------------------------------------------------------------------
def test_try_wrap_rope_finds_rope_apply(monkeypatch):
    """_try_wrap_rope finds rope_apply in a freshly injected module."""
    _reset(monkeypatch)

    def fake_rope_apply(x, grid_sizes, freqs):
        return x

    fake_mod = types.ModuleType("_test_fake_rope_mod_a")
    fake_mod.rope_apply = fake_rope_apply
    monkeypatch.setitem(sys.modules, "_test_fake_rope_mod_a", fake_mod)

    mps_profile._try_wrap_rope()

    assert getattr(fake_mod.rope_apply, "_asfp8_timed", False), (
        "rope_apply should have been replaced with a _timed('rotary', ...) wrapper"
    )
    assert len(mps_profile._rope_wrapped_ids) > 0, (
        "_rope_wrapped_ids should be non-empty after wrapping rope_apply"
    )


def test_try_wrap_rope_finds_apply_rope(monkeypatch):
    """Scanner finds apply_rope (Flux / Chroma / Ideogram naming convention)."""
    _reset(monkeypatch)

    def fake_apply_rope(xq, xk, freqs_cis):
        return xq, xk

    fake_mod = types.ModuleType("_test_fake_rope_mod_b")
    fake_mod.apply_rope = fake_apply_rope
    monkeypatch.setitem(sys.modules, "_test_fake_rope_mod_b", fake_mod)

    mps_profile._try_wrap_rope()
    assert getattr(fake_mod.apply_rope, "_asfp8_timed", False), (
        "apply_rope should have been replaced with a _timed wrapper"
    )


def test_try_wrap_rope_finds_apply_rotary_emb(monkeypatch):
    """Scanner finds apply_rotary_emb (HunyuanVideo / LTX naming)."""
    _reset(monkeypatch)

    def fake_are(x, freqs_cis):
        return x

    fake_mod = types.ModuleType("_test_fake_rope_mod_c")
    fake_mod.apply_rotary_emb = fake_are
    monkeypatch.setitem(sys.modules, "_test_fake_rope_mod_c", fake_mod)

    mps_profile._try_wrap_rope()
    assert getattr(fake_mod.apply_rotary_emb, "_asfp8_timed", False)


def test_try_wrap_rope_finds_ideogram4_apply_rope_lowp(monkeypatch):
    """Scanner finds _ideogram4_apply_rope_lowp (KJNodes Ideogram4 RoPE path).

    Without this name in _ROPE_FN_NAMES, the Ideogram4 convrot run would
    report rotary = 0% even though RoPE ran inside the KJNodes kernel.
    """
    _reset(monkeypatch)

    def fake_ideogram4_rope(xq, xk, freqs, rope_2d):
        return xq, xk

    fake_mod = types.ModuleType("_test_fake_rope_kjnodes")
    fake_mod._ideogram4_apply_rope_lowp = fake_ideogram4_rope
    monkeypatch.setitem(sys.modules, "_test_fake_rope_kjnodes", fake_mod)

    mps_profile._try_wrap_rope()
    assert getattr(fake_mod._ideogram4_apply_rope_lowp, "_asfp8_timed", False), (
        "_ideogram4_apply_rope_lowp must be wrapped (KJNodes Ideogram4 RoPE path)"
    )


# ---------------------------------------------------------------------------
# 5. Lazy RoPE scanner — alias/identity scanning (Pass 2)
# ---------------------------------------------------------------------------
def test_try_wrap_rope_patches_aliases_by_identity(monkeypatch):
    """Pass 2 patches all modules holding the same function under an alias name.

    Simulates: wanvideo/modules/model.py does
        from ... import rope_apply as apply_rope_comfy1
    then calls apply_rope_comfy1 at lines 441-451.
    Name-only scanning would leave apply_rope_comfy1 unwrapped.
    Identity scanning (Pass 2) must patch it too.
    """
    _reset(monkeypatch)

    def fake_rope_apply(x, freqs):
        return x

    source_mod = types.ModuleType("_test_rope_source")
    source_mod.rope_apply = fake_rope_apply
    monkeypatch.setitem(sys.modules, "_test_rope_source", source_mod)

    caller_mod = types.ModuleType("_test_rope_caller")
    caller_mod.apply_rope_comfy1 = fake_rope_apply  # same object, alias name
    monkeypatch.setitem(sys.modules, "_test_rope_caller", caller_mod)

    mps_profile._try_wrap_rope()

    assert getattr(source_mod.rope_apply, "_asfp8_timed", False), (
        "source module: rope_apply must be wrapped (Pass 1)"
    )
    assert getattr(caller_mod.apply_rope_comfy1, "_asfp8_timed", False), (
        "caller module: alias apply_rope_comfy1 must be wrapped by identity (Pass 2)"
    )


# ---------------------------------------------------------------------------
# 6. Lazy RoPE scanner — deduplication and edge cases
# ---------------------------------------------------------------------------
def test_try_wrap_rope_skips_known_id(monkeypatch):
    """A function whose id() is in _rope_wrapped_ids is not re-wrapped."""
    _reset(monkeypatch)

    def fake_rope(x, freqs):
        return x

    original_fn = fake_rope
    monkeypatch.setattr(mps_profile, "_rope_wrapped_ids", {id(fake_rope)})

    fake_mod = types.ModuleType("_test_fake_rope_mod_d")
    fake_mod.rope_apply = fake_rope
    monkeypatch.setitem(sys.modules, "_test_fake_rope_mod_d", fake_mod)

    mps_profile._try_wrap_rope()
    assert fake_mod.rope_apply is original_fn, (
        "function whose id is in _rope_wrapped_ids must not be re-wrapped"
    )


def test_try_wrap_rope_no_double_wrap(monkeypatch):
    """A function already carrying _asfp8_timed=True must not be re-wrapped."""
    _reset(monkeypatch)

    def fake_rope(x, freqs):
        return x

    fake_rope._asfp8_timed = True
    original_fn = fake_rope

    fake_mod = types.ModuleType("_test_fake_rope_mod_e")
    fake_mod.rope_apply = fake_rope
    monkeypatch.setitem(sys.modules, "_test_fake_rope_mod_e", fake_mod)

    mps_profile._try_wrap_rope()
    assert fake_mod.rope_apply is original_fn, (
        "already-wrapped function must not be replaced"
    )


def test_try_wrap_rope_skips_callable_without_code(monkeypatch):
    """A callable without __code__ (functools.partial, builtin) must be skipped."""
    _reset(monkeypatch)

    fake_rope = functools.partial(lambda x: x)
    assert not hasattr(fake_rope, "__code__"), "precondition: partial has no __code__"

    fake_mod = types.ModuleType("_test_fake_rope_mod_f")
    fake_mod.rope_apply = fake_rope
    monkeypatch.setitem(sys.modules, "_test_fake_rope_mod_f", fake_mod)

    mps_profile._try_wrap_rope()
    assert not getattr(fake_mod.rope_apply, "_asfp8_timed", False), (
        "functools.partial should be skipped (no __code__)"
    )
    assert len(mps_profile._rope_wrapped_ids) == 0, (
        "_rope_wrapped_ids must remain empty when only unqualified callables found"
    )


def test_try_wrap_rope_handles_varargs_function(monkeypatch):
    """*args-only functions (co_argcount == 0) with __code__ must be wrapped.

    The original co_argcount < 1 guard was wrong: valid RoPE wrappers that
    use only *args have co_argcount == 0 but have __code__. The correct guard
    is __code__ is None (builtins/C-extensions), not co_argcount.
    """
    _reset(monkeypatch)

    def fake_rope_varargs(*args):
        return args[0] if args else None

    assert fake_rope_varargs.__code__.co_argcount == 0, (
        "precondition: *args-only function has co_argcount == 0"
    )

    fake_mod = types.ModuleType("_test_fake_rope_varargs")
    fake_mod.rope_apply = fake_rope_varargs
    monkeypatch.setitem(sys.modules, "_test_fake_rope_varargs", fake_mod)

    mps_profile._try_wrap_rope()
    assert getattr(fake_mod.rope_apply, "_asfp8_timed", False), (
        "*args-only function with __code__ should be wrapped (co_argcount guard removed)"
    )


def test_try_wrap_rope_does_not_wrap_when_no_targets(monkeypatch):
    """_rope_wrapped_ids does not grow when no rope-named callables exist.

    Injects a decoy module with non-rope attribute names, then asserts that
    neither the decoy function nor its id appears in _rope_wrapped_ids.
    """
    _reset(monkeypatch)

    original_fn = lambda x: x  # noqa: E731
    decoy_mod = types.ModuleType("_test_decoy_no_rope_xyz")
    decoy_mod.unrelated_function = original_fn
    monkeypatch.setitem(sys.modules, "_test_decoy_no_rope_xyz", decoy_mod)

    mps_profile._try_wrap_rope()

    # Decoy must be untouched
    assert decoy_mod.unrelated_function is original_fn, (
        "non-rope-named function must not be wrapped"
    )
    assert id(original_fn) not in mps_profile._rope_wrapped_ids, (
        "non-target function id must not appear in _rope_wrapped_ids"
    )


# ---------------------------------------------------------------------------
# 7. install() guard — no-op without ASFP8_PROFILE=1
# ---------------------------------------------------------------------------
def test_install_noop_without_env(monkeypatch):
    """install() must be a no-op when ASFP8_PROFILE env var is absent or not '1'."""
    monkeypatch.delenv("ASFP8_PROFILE", raising=False)
    monkeypatch.setattr(mps_profile, "_installed", False)
    mps_profile.install()
    assert mps_profile._installed is False, (
        "install() should not set _installed without ASFP8_PROFILE=1"
    )
