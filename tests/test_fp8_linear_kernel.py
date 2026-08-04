import os
import shutil
import threading
import time

import pytest
import torch

from _patches.fp8_ext import loader


def _reset_loader(monkeypatch):
    monkeypatch.setattr(loader, "_tried", False, raising=False)
    monkeypatch.setattr(loader, "_mod", None, raising=False)


def test_loader_gated_off_when_unsupported(monkeypatch):
    # DEFAULT ON but capability-gated: flags unset + unsupported HW -> no build.
    from _patches import _caps
    monkeypatch.delenv("ASFP8_FP8_EXT", raising=False)
    monkeypatch.delenv("ASFP8_FP8_NATIVE", raising=False)
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: False)
    _reset_loader(monkeypatch)
    assert loader.module() is None


def test_loader_forced_off_beats_capability(monkeypatch):
    # Explicit off on both flags wins even on capable HW -> no build.
    from _patches import _caps
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: True)
    monkeypatch.setenv("ASFP8_FP8_EXT", "off")
    monkeypatch.setenv("ASFP8_FP8_NATIVE", "off")
    _reset_loader(monkeypatch)
    assert loader.module() is None


def test_loader_memo_resets_between_flag_states(monkeypatch):
    # Off -> None, then flip NATIVE on with a fresh reset; the gate must re-evaluate.
    from _patches import _caps
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: False)
    monkeypatch.delenv("ASFP8_FP8_EXT", raising=False)
    monkeypatch.delenv("ASFP8_FP8_NATIVE", raising=False)
    _reset_loader(monkeypatch)
    assert loader.module() is None
    monkeypatch.setenv("ASFP8_FP8_NATIVE", "1")
    _reset_loader(monkeypatch)
    # On non-MPS/no-toolchain CI this returns None via a *different* branch (xcrun/build),
    # NOT the env gate; that is still correct. On this M5 it builds and returns a module.
    _ = loader.module()  # must not raise; value depends on host


# --- Task 3: unit guards (no kernel needed; run everywhere) ---------------------
from _patches import fp8_linear_kernel_mps as patch


def test_install_noop_when_explicitly_off(monkeypatch):
    # ASFP8_FP8_NATIVE=off force-disables even on capable hardware.
    from _patches import _caps
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: True)
    monkeypatch.setenv("ASFP8_FP8_NATIVE", "off")
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


def test_install_noop_when_not_capable(monkeypatch):
    # DEFAULT ON but Tier-B gated: unsupported hardware -> no build, stays inert.
    from _patches import _caps
    monkeypatch.delenv("ASFP8_FP8_NATIVE", raising=False)
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: False)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


# --- the Metal build must never run on the ComfyUI startup thread ---------------


def test_install_does_not_build_the_extension(monkeypatch):
    """install() runs at import time: it may wire the seam, never build the kernel."""
    from _patches import _caps
    monkeypatch.delenv("ASFP8_FP8_NATIVE", raising=False)
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: True)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    monkeypatch.setattr(patch, "_kernel_tried", False, raising=False)

    builds = []
    monkeypatch.setattr(patch, "_load_kernel", lambda: builds.append(1), raising=False)

    patch.install()
    assert builds == [], "install() built the Metal extension on the startup thread"


def test_ineligible_layer_does_not_trigger_a_build(monkeypatch):
    """A non-fp8 Linear must not drag in the multi-minute Metal build."""
    builds = []
    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    monkeypatch.setattr(patch, "_kernel_tried", False, raising=False)
    monkeypatch.setattr(patch, "_load_kernel", lambda: builds.append(1), raising=False)

    class Holder:
        weight = torch.zeros(8, 8)  # plain tensor, not a QuantizedTensor
        bias = None

    assert patch._try_fp8_kernel_forward(Holder(), torch.zeros(4, 8)) is None
    assert builds == [], "an ineligible layer triggered the Metal build"


def test_loader_gives_up_when_the_build_stalls(monkeypatch):
    """A stalled toolchain must degrade to 'kernel unavailable', not block forever."""
    import torch.utils.cpp_extension as cpp

    from _patches import _caps

    if shutil.which("xcrun") is None or not _caps.ninja_available():
        pytest.skip("needs the Metal toolchain + ninja to reach the build call")

    monkeypatch.delenv("ASFP8_FP8_EXT", raising=False)
    monkeypatch.delenv("ASFP8_FP8_NATIVE", raising=False)
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: True)
    monkeypatch.setenv("ASFP8_EXT_BUILD_TIMEOUT", "0.5")
    _reset_loader(monkeypatch)

    started = threading.Event()
    never = object()

    def stalled_build(*a, **k):
        started.set()
        time.sleep(5.0)
        return never

    monkeypatch.setattr(cpp, "load", stalled_build)

    t0 = time.monotonic()
    mod = loader.module()
    elapsed = time.monotonic() - t0

    assert started.is_set(), "the build was never attempted"
    assert mod is None, "a stalled build must report the kernel as unavailable"
    assert elapsed < 3.0, f"loader blocked {elapsed:.1f}s on a stalled build"


def test_eligibility_returns_none_no_kernel(monkeypatch):
    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    class FakeLinear: weight = torch.zeros(8, 8, dtype=torch.float8_e4m3fn)
    assert patch._try_fp8_kernel_forward(FakeLinear(), torch.zeros(4, 8)) is None


def test_eligibility_rejects_non_fp8_weight(monkeypatch):
    # Kernel present but weight isn't a fp8 QuantizedTensor -> None (FAST handback to int8/orig).
    monkeypatch.setattr(patch, "_kernel", object(), raising=False)
    class FakeLinear: weight = torch.zeros(8, 8)   # plain, not QuantizedTensor
    assert patch._try_fp8_kernel_forward(FakeLinear(), torch.zeros(4, 8, device="cpu")) is None


def test_self_check_not_run_on_ineligible(monkeypatch):
    # BLOCKER 2 regression guard: a non-MPS/non-fp8 layer must NOT trigger the self-check.
    monkeypatch.setattr(patch, "_kernel", object(), raising=False)
    monkeypatch.setattr(patch, "_self_checked", False, raising=False)
    calls = {"n": 0}
    def boom():
        calls["n"] += 1
        return True
    monkeypatch.setattr(patch, "_self_check", boom, raising=False)
    class FakeLinear: weight = torch.zeros(8, 8)   # ineligible
    assert patch._try_fp8_kernel_forward(FakeLinear(), torch.zeros(4, 8, device="cpu")) is None
    assert calls["n"] == 0, "self-check ran on an ineligible layer (BLOCKER 2 regression)"


# --- Task 4: REAL spy tests (kernel really ran AND wrapper dispatched native) -----
# Gated on ASFP8_FP8_NATIVE=1 + MPS (builds the Metal lib).
_run = os.environ.get("ASFP8_FP8_NATIVE") == "1"
requires_fp8_native = pytest.mark.skipif(
    not (_run and torch.backends.mps.is_available()),
    reason="set ASFP8_FP8_NATIVE=1 on an MPS device to build + test the fp8 kernel")

# MPS cannot cast bf16->fp8, so real fp8 QuantizedTensors are built on CPU then moved.
# This comfy_kitchen registers the e4m3 layout as "TensorCoreFP8Layout".
_FP8_LAYOUT = "TensorCoreFP8Layout"


@requires_fp8_native
def test_native_matches_ground_truth(monkeypatch):
    from _patches.fp8_ext import loader
    from _patches._common import decode_fp8
    mod = loader.module(); assert mod is not None; mod.warmup()
    monkeypatch.setattr(patch, "_kernel", mod, raising=False)

    M, K, N = 64, 8192, 8192
    g = torch.Generator().manual_seed(3)
    act = (torch.randn(M, K, generator=g) * 0.3).to(torch.bfloat16).to("mps")
    w_fp8 = (torch.randn(N, K, generator=g) * 0.3).to(torch.float8_e4m3fn).to("mps")
    scale = torch.tensor(1.0)

    out = patch._fp8_linear_kernel(act, w_fp8, scale, None)
    gt = (act.float() @ decode_fp8(w_fp8, torch.float32).t()).to(torch.bfloat16)  # all-MPS fp32 ref
    assert out.dtype == torch.bfloat16
    rel = ((out.float() - gt.float()).abs().max() / (gt.float().abs().max() + 1e-9)).item()
    assert rel < 2e-2, f"native vs ground-truth rel={rel}"
    assert out.abs().max() > 0  # SPY: not all-zero


@requires_fp8_native
def test_native_per_channel_scale(monkeypatch):
    # Exercise the [N] scale branch (resolves OQ#1 in code regardless of Task -1 finding).
    from _patches.fp8_ext import loader
    from _patches._common import decode_fp8
    mod = loader.module(); assert mod is not None; mod.warmup()
    monkeypatch.setattr(patch, "_kernel", mod, raising=False)
    M, K, N = 64, 8192, 8192
    g = torch.Generator().manual_seed(5)
    act = (torch.randn(M, K, generator=g) * 0.3).to(torch.bfloat16).to("mps")
    w_fp8 = (torch.randn(N, K, generator=g) * 0.3).to(torch.float8_e4m3fn).to("mps")
    scale = (torch.rand(N, generator=g) * 0.5 + 0.5)  # [N]
    out = patch._fp8_linear_kernel(act, w_fp8, scale, None)
    gt = (act.float() @ decode_fp8(w_fp8, torch.float32).t()) * scale.to("mps").reshape(1, N)
    rel = ((out.float() - gt.float()).abs().max() / (gt.abs().max() + 1e-9)).item()
    assert rel < 2e-2, f"per-channel native rel={rel}"


@requires_fp8_native
def test_wrapper_dispatches_native_not_fallback(monkeypatch):
    """End-to-end SPY: build a REAL QuantizedTensor fp8 weight, monkeypatch
    _fp8_linear_kernel to a sentinel AND the captured orig_forward to RAISE.
    If Linear.forward returns the sentinel, the native path dispatched; if the fallback
    ran, the test ERRORS (orig_forward raises) instead of silently passing."""
    from comfy_kitchen.tensor import QuantizedTensor
    from _patches.fp8_ext import loader
    mod = loader.module(); assert mod is not None
    monkeypatch.setattr(patch, "_kernel", mod, raising=False)
    monkeypatch.setattr(patch, "_self_checked", True, raising=False)
    monkeypatch.setattr(patch, "_self_ok", True, raising=False)

    SENTINEL = torch.full((1,), 1234.0)
    monkeypatch.setattr(patch, "_fp8_linear_kernel",
                        lambda *a, **k: SENTINEL, raising=False)

    # Build a real fp8 QuantizedTensor weight on CPU (MPS can't cast bf16->fp8), then
    # move to MPS so _qdata is MPS-resident fp8 e4m3 (layout_cls starts "TensorCoreFP8").
    N, K = 8192, 8192
    wf = (torch.randn(N, K) * 0.3).to(torch.bfloat16)
    qw = QuantizedTensor.from_float(wf, _FP8_LAYOUT, scale=torch.tensor(1.0))
    qw = qw.to("mps")
    assert qw._qdata.is_mps and qw._qdata.dtype == torch.float8_e4m3fn
    assert str(qw._layout_cls).startswith("TensorCoreFP8")

    # Minimal Linear-like holder matching the attributes the eligibility reads.
    class Holder:
        weight = qw
        bias = None
        _full_precision_mm = False
        comfy_force_cast_weights = False
        weight_function = []
        bias_function = []
    holder = Holder()

    # Construct the patched forward exactly as install() does, with a RAISING orig_forward.
    def orig_forward(self, input, *a, **k):
        raise AssertionError("fallback orig_forward ran; native path did NOT dispatch")

    def forward(self, input, *a, **k):
        res = patch._try_fp8_kernel_forward(self, input)
        return res if res is not None else orig_forward(self, input, *a, **k)

    inp = (torch.randn(64, K) * 0.3).to(torch.bfloat16).to("mps")
    out = forward(holder, inp)
    assert torch.equal(out, SENTINEL), "wrapper did not dispatch the native sentinel path"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_failed_build_short_circuits_before_the_range_guard(monkeypatch):
    """Once the build is known to have failed, an eligible layer must bail early.

    The fp16 range guard is a full-tensor amax plus a device sync, and it sits
    ahead of _ensure_kernel(), so without a top-level short circuit every
    eligible fp8 Linear pays it on every step for the rest of the session.
    """
    QuantizedTensor = pytest.importorskip("comfy_kitchen.tensor").QuantizedTensor
    from _patches import fp8_linear_kernel_mps as patch

    monkeypatch.setenv("ASFP8_FP8_NATIVE_MIN_DIM", "8")
    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    monkeypatch.setattr(patch, "_kernel_tried", True, raising=False)

    wf = torch.randn(32, 32, dtype=torch.bfloat16)
    qw = QuantizedTensor.from_float(
        wf, "TensorCoreFP8Layout", scale=torch.tensor(1.0)
    ).to(device="mps")

    class Layer:
        weight = qw
        bias = None
        weight_function = []
        bias_function = []

    x = torch.randn(8, 32, dtype=torch.bfloat16, device="mps")

    seen = []
    real_amax = torch.Tensor.amax
    monkeypatch.setattr(
        torch.Tensor,
        "amax",
        lambda self, *a, **k: (seen.append(1), real_amax(self, *a, **k))[1],
    )

    assert patch._try_fp8_kernel_forward(Layer(), x) is None
    assert seen == [], "range guard ran though the kernel is known unavailable"
