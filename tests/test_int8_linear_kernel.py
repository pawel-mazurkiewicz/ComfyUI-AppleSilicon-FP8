"""Tests for patch #17: INT8 W8A8 via the bit-exact Metal kernel.

The integration tests need ASFP8_INT8_EXT=1 on an MPS box; the rest always run.
"""
import os
import shutil
import threading
import time

import pytest
import torch

from conftest import requires_mps

from _patches import int8_linear_kernel_mps as patch

from _patches import _caps

# the same gate production uses: behind an opt-in env var instead, a kernel that
# stopped compiling showed up as silent skips
_int8_enabled = torch.backends.mps.is_available() and _caps.resolve(
    "ASFP8_INT8_EXT", default_on=True, cap=_caps.kernel_gate
)


def _int8_kernel_works():
    """Build the extension and run its self-check once, at collection time."""
    if not _int8_enabled:
        return False
    try:
        return patch._ensure_kernel() is not None and patch._self_check()
    except Exception:
        return False


_int8_ok = _int8_kernel_works()

requires_int8_ext = pytest.mark.skipif(
    not _int8_ok,
    reason="int8 kernel unavailable here — see test_int8_kernel_compiles_when_enabled",
)


@pytest.fixture(autouse=True)
def _clear_int8_kernel_memo():
    """Clear _caps.kernel_ready's per-process memo, or the first test to verify the
    kernel decides the answer for every later one."""
    from _patches import _caps
    _caps._kernel_ready.pop("int8", None)
    yield
    _caps._kernel_ready.pop("int8", None)


def test_install_noop_when_explicitly_off(monkeypatch):
    """ASFP8_INT8_EXT=off force-disables even on capable hardware."""
    from _patches import _caps
    monkeypatch.setattr(_caps, "kernel_gate", lambda: True)
    monkeypatch.setenv("ASFP8_INT8_EXT", "off")
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


def test_install_noop_when_not_capable(monkeypatch):
    """DEFAULT ON but gated: unsupported hardware -> no build attempt, stays inert."""
    from _patches import _caps
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "kernel_gate", lambda: False)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


# --- the Metal build must never run on the ComfyUI startup thread ---------------


@requires_mps
def test_install_does_not_build_the_extension(monkeypatch):
    """install() runs at import time: it may wire the seam, never build the kernel."""
    from _patches import _caps
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "kernel_gate", lambda: True)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    monkeypatch.setattr(patch, "_kernel_tried", False, raising=False)

    builds = []
    monkeypatch.setattr(patch, "_load_kernel", lambda: builds.append(1), raising=False)

    patch.install()
    assert builds == [], "install() built the Metal extension on the startup thread"


@requires_mps
def test_kernel_builds_on_first_eligible_forward(monkeypatch):
    """The build is deferred to the first int8 layer that really needs it, once."""
    from comfy_kitchen.tensor import QuantizedTensor

    builds = []
    fake_kernel = object()

    def fake_build():
        builds.append(1)
        return fake_kernel

    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    monkeypatch.setattr(patch, "_kernel_tried", False, raising=False)
    monkeypatch.setattr(patch, "_load_kernel", fake_build, raising=False)
    # this test is about build deferral, not kernel correctness
    monkeypatch.setattr(patch, "_self_checked", True, raising=False)
    monkeypatch.setattr(patch, "_self_ok", True, raising=False)

    out = torch.full((1,), 7.0)
    monkeypatch.setattr(patch, "_int8_linear_kernel", lambda *a, **k: out, raising=False)

    qw = QuantizedTensor.from_float(
        (torch.randn(64, 128) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    ).to("mps")

    class Holder:
        weight = qw
        bias = None
        _full_precision_mm = False
        comfy_force_cast_weights = False
        weight_function = []
        bias_function = []

    x = torch.randn(8, 128, dtype=torch.bfloat16, device="mps")
    assert patch._try_int8_kernel_forward(Holder(), x) is out
    assert builds == [1], "first eligible forward did not build the kernel"
    assert patch._try_int8_kernel_forward(Holder(), x) is out
    assert builds == [1], "kernel was rebuilt on a later forward"


@requires_mps
def test_ineligible_layer_does_not_trigger_a_build(monkeypatch):
    """A non-int8 Linear must not drag in the multi-minute Metal build."""
    builds = []
    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    monkeypatch.setattr(patch, "_kernel_tried", False, raising=False)
    monkeypatch.setattr(patch, "_load_kernel", lambda: builds.append(1), raising=False)

    class Holder:
        weight = torch.zeros(8, 8)  # plain tensor, not a QuantizedTensor
        bias = None

    got = patch._try_int8_kernel_forward(Holder(), torch.zeros(4, 8, device="mps"))
    assert got is None
    assert builds == [], "an ineligible layer triggered the Metal build"


def test_loader_gives_up_when_the_build_stalls(monkeypatch):
    """A stalled toolchain must degrade to 'kernel unavailable', not block forever."""
    import torch.utils.cpp_extension as cpp

    from _patches import _caps
    from _patches.int8_ext import loader

    if shutil.which("xcrun") is None or not _caps.ninja_available():
        pytest.skip("needs the Metal toolchain + ninja to reach the build call")

    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "kernel_gate", lambda: True)
    monkeypatch.setenv("ASFP8_EXT_BUILD_TIMEOUT", "0.5")
    monkeypatch.setattr(loader, "_tried", False, raising=False)
    monkeypatch.setattr(loader, "_mod", None, raising=False)

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


def test_wrapper_falls_back_off_mps(monkeypatch):
    """With no kernel (or a CPU tensor) the wrapper delegates to the original."""
    sentinel = object()
    called = {}

    def fake_orig(x, w, ws, bias, out_dtype, convrot, gs):
        called["hit"] = True
        return sentinel

    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    monkeypatch.setattr(patch, "_orig_int8_linear", fake_orig, raising=False)

    x = torch.zeros(4, 8)  # CPU tensor
    w = torch.zeros(8, 8, dtype=torch.int8)
    ws = torch.ones(1)
    out = patch._int8_linear_kernel(x, w, ws)
    assert out is sentinel and called.get("hit") is True


def test_fallback_without_install_raises_clean_error(monkeypatch):
    """A direct call before install() raises a clear RuntimeError, not an AttributeError."""
    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    monkeypatch.setattr(patch, "_orig_int8_linear", None, raising=False)
    x = torch.zeros(4, 8)
    w = torch.zeros(8, 8, dtype=torch.int8)
    ws = torch.ones(1)
    with pytest.raises(RuntimeError):
        patch._int8_linear_kernel(x, w, ws)


@requires_int8_ext
def test_kernel_matches_original_bit_exact():
    """Kernel-backed int8_linear == comfy_kitchen original (bit-identical bf16)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # original _int_mm -> CPU
    from _patches.int8_ext import loader
    from comfy_kitchen.backends.eager.quantization import (
        int8_linear as orig_int8_linear,
    )

    mod = loader.module()
    assert mod is not None, "int8 kernel failed to build"
    mod.warmup()
    assert hasattr(mod, "i8_matmul2d_nt_fused"), "fused entry point missing"

    # Install so the wrapper picks up the freshly built kernel + original.
    patch._kernel = mod
    patch._orig_int8_linear = orig_int8_linear

    from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    dev = "mps"
    g = torch.Generator().manual_seed(7)

    def unfused(x, w, ws, b, convrot):
        """Chunked epilogue (int32 store + Python rescale) for cross-checking."""
        if convrot:
            h = _build_hadamard(256, device=x.device, dtype=x.dtype)
            x = _rotate_activation(x, h, 256)
        shp = x.shape
        x8, xs = quantize_int8_rowwise(x.reshape(-1, x.shape[-1]))
        C = mod.i8_matmul2d_nt(x8.contiguous(), w.contiguous()).float()
        out = (C * (ws.view(-1) * xs)).to(torch.bfloat16)
        if b is not None:
            out = out + b.to(out.dtype)
        return out.reshape(*shp[:-1], w.shape[0])

    def run(M, K, N, convrot, bias, three_d):
        shape = (2, M, K) if three_d else (M, K)
        x = (torch.randn(shape, generator=g, dtype=torch.bfloat16) * 0.5).to(dev)
        w = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to(dev)
        ws = (torch.rand(1, generator=g, dtype=torch.float32) * 0.01 + 0.001).to(dev)
        b = torch.randn(N, generator=g, dtype=torch.bfloat16).to(dev) if bias else None
        ref = orig_int8_linear(x, w, ws, b, torch.bfloat16, convrot, 256)
        # _int8_linear_kernel auto-selects the fused bf16 epilogue.
        out = patch._int8_linear_kernel(x, w, ws, b, torch.bfloat16, convrot, 256)
        assert torch.equal(ref, out), f"fused mismatch M={M} K={K} N={N} convrot={convrot}"
        # The fused epilogue must also be bit-identical to the chunked one.
        unf = unfused(x, w, ws, b, convrot)
        assert torch.equal(out, unf), f"fused!=unfused M={M} K={K} N={N} convrot={convrot}"

    run(256, 2560, 1024, convrot=False, bias=False, three_d=False)
    run(256, 2560, 1024, convrot=False, bias=True, three_d=False)
    run(512, 6144, 6144, convrot=True, bias=False, three_d=False)
    run(512, 6144, 6144, convrot=True, bias=True, three_d=False)
    run(188, 4096, 2560, convrot=True, bias=True, three_d=True)
    run(1, 6144, 6144, convrot=True, bias=False, three_d=False)


# gelu-erf is absent (Metal has no `erf`), so only {silu, gelu_tanh} are supported
@requires_int8_ext
@pytest.mark.parametrize("act", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("convrot", [False, True])  # rotation shifts the activation magnitude dist
@pytest.mark.parametrize("M", [1, 256])
def test_int8_linear_fused_activation_matches_reference(M, convrot, bias, act):
    """Fused-epilogue activation == torch activation of the unfused kernel output."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from _patches.int8_ext import loader
    from comfy_kitchen.backends.eager.quantization import int8_linear as orig_int8_linear
    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_matmul2d_nt_fused")
    mod.warmup()
    patch._kernel = mod
    patch._orig_int8_linear = orig_int8_linear

    dev = "mps"
    g = torch.Generator().manual_seed(11)
    K, N = 2560, 1024
    x = (torch.randn(M, K, generator=g, dtype=torch.bfloat16) * 0.5).to(dev)
    w = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to(dev)
    ws = (torch.rand(1, generator=g, dtype=torch.float32) * 0.01 + 0.001).to(dev)
    b = torch.randn(N, generator=g, dtype=torch.bfloat16).to(dev) if bias else None

    lin = patch._int8_linear_kernel(x, w, ws, b, torch.bfloat16, convrot, 256, act="none")
    if act == "silu":
        ref = torch.nn.functional.silu(lin)
    else:
        ref = torch.nn.functional.gelu(lin, approximate="tanh")

    # Spy guard: the fused activation path must be the real kernel, not the torch fallback.
    def _boom(*a, **k):
        raise AssertionError("fell back to _orig_int8_linear; fused kernel did not run")
    saved = patch._orig_int8_linear
    patch._orig_int8_linear = _boom
    try:
        out = patch._int8_linear_kernel(x, w, ws, b, torch.bfloat16, convrot, 256, act=act)
    finally:
        patch._orig_int8_linear = saved
    torch.mps.synchronize()
    assert out.shape == ref.shape
    # rtol is one bf16 ulp: both sides round to bf16, so anything tighter is below
    # bf16 precision and unsatisfiable. atol bounds the near-zero regime.
    d = (out.float() - ref.float()).abs()
    assert torch.allclose(out, ref, atol=2e-3, rtol=8e-3), \
        f"M={M} convrot={convrot} bias={bias} {act}: max|d|={d.max().item():.4g}"


def test_wrapper_fallback_applies_activation(monkeypatch):
    """The no-kernel fallback still applies the requested activation."""
    monkeypatch.setattr(patch, "_kernel", None)  # force the early fallback branch

    captured = {}
    def fake_orig(x, w, ws, bias, out_dtype, convrot, gs):
        captured["called"] = True
        return torch.full((x.shape[0], w.shape[0]), 2.0, dtype=torch.float32)
    monkeypatch.setattr(patch, "_orig_int8_linear", fake_orig)

    x = torch.randn(4, 8)
    w = torch.randint(-128, 128, (3, 8), dtype=torch.int8)
    ws = torch.tensor([0.01])
    out = patch._int8_linear_kernel(x, w, ws, None, torch.float32, False, 256, act="silu")
    assert captured.get("called"), "fallback path was not taken"
    # silu(2.0) ≈ 1.7616, not the raw 2.0 — proves the activation was applied post-fallback.
    assert torch.allclose(out, torch.nn.functional.silu(torch.full_like(out, 2.0)))


def test_wrapper_rejects_unknown_act():
    with pytest.raises(ValueError):
        patch._int8_linear_kernel(torch.randn(2, 4), torch.randint(-1, 2, (3, 4),
                                  dtype=torch.int8), torch.tensor([0.01]), None,
                                  torch.float32, False, 256, act="sillu")


def test_swiglu_rejects_unknown_act():
    """A typo'd activation raises before any dispatch, for the gated kernel too."""
    w = torch.randint(-1, 2, (3, 4), dtype=torch.int8)
    s = torch.tensor([0.01])
    with pytest.raises(ValueError):
        patch._int8_swiglu_kernel(torch.randn(2, 4), w, w, s, s,
                                  None, None, False, 256, act="sillu")


def test_swiglu_rejects_none_act():
    """act="none" is rejected for a gate: it would degenerate to an elementwise product."""
    w = torch.randint(-1, 2, (3, 4), dtype=torch.int8)
    s = torch.tensor([0.01])
    with pytest.raises(ValueError):
        patch._int8_swiglu_kernel(torch.randn(2, 4), w, w, s, s,
                                  None, None, False, 256, act="none")


# tolerance is one bf16 ulp, as above. K=2576 covers remainder-K; the convrot case
# covers the Hadamard-rotate -> requant path into the fused gate kernel.
@requires_int8_ext
@pytest.mark.parametrize("act", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "M,K,convrot",
    [(1, 2560, False), (512, 2560, False), (512, 2576, False), (256, 2560, True)],
)
def test_int8_swiglu_matches_reference(M, K, convrot, bias, act):
    """Fused gate == act(int8_linear(x,Wg)) * int8_linear(x,Wu)."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from _patches.int8_ext import loader
    from comfy_kitchen.backends.eager.quantization import int8_linear as orig_int8_linear
    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_matmul2d_nt_swiglu")
    mod.warmup()
    patch._kernel = mod
    patch._orig_int8_linear = orig_int8_linear

    dev = "mps"
    g = torch.Generator().manual_seed(5)
    N = 1024
    x  = (torch.randn(M, K, generator=g, dtype=torch.bfloat16) * 0.5).to(dev)
    wg = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to(dev)
    wu = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to(dev)
    sg = (torch.rand(1, generator=g, dtype=torch.float32) * 0.01 + 0.001).to(dev)
    su = (torch.rand(1, generator=g, dtype=torch.float32) * 0.01 + 0.001).to(dev)
    bg = torch.randn(N, generator=g, dtype=torch.bfloat16).to(dev) if bias else None
    bu = torch.randn(N, generator=g, dtype=torch.bfloat16).to(dev) if bias else None

    lin_g = patch._int8_linear_kernel(x, wg, sg, bg, torch.bfloat16, convrot, 256, act="none")
    lin_u = patch._int8_linear_kernel(x, wu, su, bu, torch.bfloat16, convrot, 256, act="none")
    # the fused gate rounds to bf16 ONCE, so the reference must too: a bf16-rounded
    # intermediate gate would double-round and diverge, which is not a kernel bug
    gfp = lin_g.float()
    gate = torch.nn.functional.silu(gfp) if act == "silu" \
           else torch.nn.functional.gelu(gfp, approximate="tanh")
    ref = (gate * lin_u.float()).to(torch.bfloat16)

    # Spy guard: prove the fused gated kernel ran, not the per-branch fallback.
    def _boom(*a, **k):
        raise AssertionError("SwiGLU fell back to _orig_int8_linear; fused gate did not run")
    saved = patch._orig_int8_linear
    patch._orig_int8_linear = _boom
    try:
        out = patch._int8_swiglu_kernel(x, wg, wu, sg, su, bg, bu, convrot, 256, act=act)
    finally:
        patch._orig_int8_linear = saved
    torch.mps.synchronize()
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=2e-3, rtol=8e-3), \
        f"M={M} K={K} convrot={convrot} {act} bias={bias}: max|d|={(out.float()-ref.float()).abs().max().item():.4g}"


@requires_int8_ext
def test_int8_swiglu_nonscalar_scale_falls_back_correctly():
    """Length-N weight scales route through the per-branch path, not the fused gate."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from _patches.int8_ext import loader
    from comfy_kitchen.backends.eager.quantization import int8_linear as orig_int8_linear
    mod = loader.module()
    assert mod is not None
    mod.warmup()
    patch._kernel = mod
    patch._orig_int8_linear = orig_int8_linear

    dev = "mps"
    g = torch.Generator().manual_seed(7)
    M, K, N = 64, 2560, 1024
    x  = (torch.randn(M, K, generator=g, dtype=torch.bfloat16) * 0.5).to(dev)
    wg = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to(dev)
    wu = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to(dev)
    sg = (torch.rand(N, generator=g, dtype=torch.float32) * 0.01 + 0.001).to(dev)  # per-channel
    su = (torch.rand(N, generator=g, dtype=torch.float32) * 0.01 + 0.001).to(dev)

    lin_g = patch._int8_linear_kernel(x, wg, sg, None, torch.bfloat16, False, 256, act="none")
    lin_u = patch._int8_linear_kernel(x, wu, su, None, torch.bfloat16, False, 256, act="none")
    ref = torch.nn.functional.silu(lin_g) * lin_u
    out = patch._int8_swiglu_kernel(x, wg, wu, sg, su, None, None, False, 256, act="silu")
    torch.mps.synchronize()
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=2e-3, rtol=8e-3), \
        f"nonscalar: max|d|={(out.float()-ref.float()).abs().max().item():.4g}"


@requires_mps
def test_metal_compile_failure_is_latched_not_retried(monkeypatch):
    """A Metal library the toolchain rejects on first use is latched off, not recompiled
    on every eligible Linear (#13)."""
    from comfy_kitchen.tensor import QuantizedTensor

    calls = []

    class FailingKernel:
        def i8_matmul2d_nt(self, *args, **kwargs):
            calls.append(1)
            raise RuntimeError(
                "int8 Metal library compile failed: no matching member function "
                "for call to 'get_destination_cooperative_tensor'"
            )

    monkeypatch.setattr(patch, "_kernel", FailingKernel(), raising=False)
    monkeypatch.setattr(patch, "_kernel_tried", True, raising=False)
    monkeypatch.setattr(patch, "_self_checked", False, raising=False)
    monkeypatch.setattr(patch, "_self_ok", False, raising=False)

    qw = QuantizedTensor.from_float(
        (torch.randn(64, 128) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    ).to("mps")

    class Holder:
        weight = qw
        bias = None
        _full_precision_mm = False
        comfy_force_cast_weights = False
        weight_function = []
        bias_function = []

    x = torch.randn(8, 128, dtype=torch.bfloat16, device="mps")

    assert patch._try_int8_kernel_forward(Holder(), x) is None
    assert patch._try_int8_kernel_forward(Holder(), x) is None

    assert len(calls) == 1, f"Metal library compile retried per forward ({len(calls)}x)"


@pytest.mark.skipif(not _int8_enabled, reason="int8 kernel not enabled on this machine")
def test_int8_kernel_compiles_when_enabled():
    """Canary: an int8 kernel the node turns on actually works (#13).

    Without it, an OS or toolchain update that kills the Metal library only shows up
    as the rest of these tests skipping.
    """
    assert patch._ensure_kernel() is not None, "int8 cpp_extension failed to build"
    assert patch._self_check(), (
        "int8 Metal library does not compile on this machine — the kernel is "
        "inert and every eligible Linear falls back to comfy's int8 path"
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_failed_verification_disables_the_kernel_and_is_not_retried(monkeypatch):
    """A kernel that fails verification costs one attempt, not one per layer (#14)."""
    from comfy_kitchen.tensor import QuantizedTensor
    from _patches import _caps

    attempts = []

    def failing_verify():
        attempts.append(1)
        return False

    monkeypatch.setattr(patch, "_verify", failing_verify)

    qw = QuantizedTensor.from_float(
        (torch.randn(64, 128) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    ).to("mps")

    class Holder:
        weight = qw
        bias = None
        _full_precision_mm = False
        comfy_force_cast_weights = False
        weight_function = []
        bias_function = []

    x = torch.randn(8, 128, dtype=torch.bfloat16, device="mps")

    for _ in range(3):
        assert patch._try_int8_kernel_forward(Holder(), x) is None

    assert len(attempts) == 1, f"verification retried per forward ({len(attempts)}x)"
    assert _caps._kernel_ready["int8"] is False


@requires_mps
def test_a_forward_failure_after_verification_disables_the_kernel(monkeypatch):
    """A dispatch failure after verification passed is latched off for the session (#13)."""
    from comfy_kitchen.tensor import QuantizedTensor
    from _patches import _caps

    calls = []

    def exploding_kernel(*a, **k):
        calls.append(1)
        raise RuntimeError("dispatch blew up")

    monkeypatch.setattr(patch, "_verify", lambda: True)
    monkeypatch.setattr(patch, "_int8_linear_kernel", exploding_kernel)

    qw = QuantizedTensor.from_float(
        (torch.randn(64, 128) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    ).to("mps")

    class Holder:
        weight = qw
        bias = None
        _full_precision_mm = False
        comfy_force_cast_weights = False
        weight_function = []
        bias_function = []

    x = torch.randn(8, 128, dtype=torch.bfloat16, device="mps")
    for _ in range(4):
        assert patch._try_int8_kernel_forward(Holder(), x) is None

    assert len(calls) == 1, f"kernel re-entered after a dispatch failure ({len(calls)}x)"
    assert _caps._kernel_ready["int8"] is False


# --- the hardware gate: issues #25 and #27 ----------------------------------
# One defect from both ends: the old gate measured the torch build's default MSL
# rather than the GPU, so it answered yes on an M4 Pro and no on an M5 Max.


def test_install_is_inert_on_a_pre_m5_chip(monkeypatch):
    """A chip positively named as pre-M5 never reaches the build at all (#25)."""
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: "Apple M4 Pro")
    monkeypatch.setattr(_caps, "ninja_available", lambda: True)
    monkeypatch.setattr(_caps, "is_mps", lambda: True)
    monkeypatch.setattr(patch, "_installed", False, raising=False)

    patch.install()

    assert patch._installed is False


def _stub_comfy_ops(monkeypatch):
    """Stand in for `comfy.ops`, which isn't importable from the repo root: without it
    install() bails on the import before the gate tests can observe anything."""
    import sys
    import types

    class _Linear:
        def forward(self, x):
            return x

    def mixed_precision_ops(*a, **k):
        return type("Ops", (), {"Linear": type("Linear", (_Linear,), {})})

    comfy = types.ModuleType("comfy")
    ops = types.ModuleType("comfy.ops")
    ops.mixed_precision_ops = mixed_precision_ops
    comfy.ops = ops
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    return ops


def test_install_survives_a_failing_compile_shader_probe(monkeypatch):
    """A failing compile_shader probe does not disable the kernel (#27)."""
    _stub_comfy_ops(monkeypatch)
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: "Apple M5 Max")
    monkeypatch.setattr(_caps, "ninja_available", lambda: True)
    monkeypatch.setattr(_caps, "is_mps", lambda: True)
    monkeypatch.setattr(_caps, "has_tensor_ops_matmul2d", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    monkeypatch.setattr(patch, "_load_kernel", lambda: None, raising=False)

    patch.install()

    assert patch._installed is True


def test_install_banner_does_not_claim_an_unverified_kernel(monkeypatch, capsys):
    """The startup banner promises a check, not a result nothing has verified yet (#25)."""
    _stub_comfy_ops(monkeypatch)
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: "Apple M5 Max")
    monkeypatch.setattr(_caps, "ninja_available", lambda: True)
    monkeypatch.setattr(_caps, "is_mps", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    monkeypatch.setattr(patch, "_load_kernel", lambda: None, raising=False)

    patch.install()

    out = capsys.readouterr().out
    assert "routed through" not in out, f"banner claims a result it hasn't got: {out}"
    assert "self-check" in out.lower(), f"banner doesn't mention verification: {out}"


def test_linear_input_act_wrapper_forwards_the_extended_signature(monkeypatch):
    """The wrapper accepts ComfyUI v0.36's extended signature (#36)."""
    ops = _stub_comfy_ops(monkeypatch)
    calls = []

    def linear_input_act(linear, x, input_act, act_weight=None, act_eps=0.0,
                         residual=None, residual_scale=None):
        calls.append((input_act, act_weight, act_eps, residual, residual_scale))
        return x

    ops.linear_input_act = linear_input_act
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: "Apple M5 Max")
    monkeypatch.setattr(_caps, "ninja_available", lambda: True)
    monkeypatch.setattr(_caps, "is_mps", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    monkeypatch.setattr(patch, "_load_kernel", lambda: None, raising=False)
    monkeypatch.setattr(patch, "_dequant_enabled", lambda: False)

    patch.install()
    assert ops.linear_input_act is not linear_input_act, "the seam was never wrapped"

    x = torch.zeros(2, 4)
    ops.linear_input_act(object(), x, "rms_norm", "norm_w", 1e-6)
    ops.linear_input_act(object(), x, "rms_norm", "norm_w", 1e-6,
                         residual="res", residual_scale="scale")
    ops.linear_input_act(object(), x, "swiglu")

    assert calls == [
        ("rms_norm", "norm_w", 1e-6, None, None),
        ("rms_norm", "norm_w", 1e-6, "res", "scale"),
        ("swiglu", None, 0.0, None, None),
    ]


@requires_mps
def test_force_cast_weights_does_not_disqualify_the_kernel(monkeypatch):
    """comfy_force_cast_weights does not push a layer off the kernel route (#26).

    On a QuantizedTensor that cast is the per-call dequant this kernel bypasses.
    """
    from comfy_kitchen.tensor import QuantizedTensor

    monkeypatch.setattr(patch, "_kernel", object(), raising=False)
    monkeypatch.setattr(patch, "_kernel_tried", True, raising=False)
    monkeypatch.setattr(patch, "_self_checked", True, raising=False)
    monkeypatch.setattr(patch, "_self_ok", True, raising=False)

    out = torch.full((1,), 7.0)
    monkeypatch.setattr(patch, "_int8_linear_kernel", lambda *a, **k: out, raising=False)

    qw = QuantizedTensor.from_float(
        (torch.randn(64, 128) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    ).to("mps")

    class Holder:
        weight = qw
        bias = None
        _full_precision_mm = False
        comfy_force_cast_weights = True
        weight_function = []
        bias_function = []

    x = torch.randn(8, 128, dtype=torch.bfloat16, device="mps")
    assert patch._try_int8_kernel_forward(Holder(), x) is out, (
        "comfy_force_cast_weights pushed an int8 layer off the kernel route"
    )


@requires_mps
def test_dequant_retries_after_a_transient_off_mps_call(monkeypatch):
    """An off-MPS first call doesn't latch _asfp8_deq_done, so a later MPS call still
    dequantises."""
    from comfy_kitchen.tensor import QuantizedTensor

    monkeypatch.setattr(patch, "_DEQUANT_MODE", True, raising=False)

    qw = QuantizedTensor.from_float(
        (torch.randn(64, 128) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    ).to("mps")

    class Holder:
        _full_precision_mm = False
        comfy_force_cast_weights = True
        weight_function = []
        bias_function = []

    layer = Holder()
    layer.weight = qw
    layer.bias = None

    patch._maybe_dequant_weight(layer, torch.randn(1, 128, dtype=torch.bfloat16))
    assert isinstance(layer.weight, QuantizedTensor), "dequantised from a CPU-input call"
    assert not getattr(layer, "_asfp8_deq_done", False), "off-MPS call latched the flag"

    x = torch.randn(1, 128, dtype=torch.bfloat16, device="mps")
    patch._maybe_dequant_weight(layer, x)
    assert not isinstance(layer.weight, QuantizedTensor), "MPS call did not dequantise"
    assert layer.weight.dtype == torch.bfloat16
    assert layer.weight.device.type == "mps"
    assert layer._asfp8_deq_done is True

    ref = qw.dequantize().to(torch.bfloat16)
    assert torch.equal(layer.weight.detach().cpu(), ref.cpu()), (
        "dequantised weight differs from QuantizedTensor.dequantize()"
    )


# --- the ASFP8_INT8_DEQUANT gate: the switch people actually touch --------------


def _resolve_dequant_gate(monkeypatch, env, total_ram):
    import psutil

    class _VM:
        total = total_ram

    monkeypatch.setattr(patch, "_DEQUANT_MODE", None, raising=False)
    if env is None:
        monkeypatch.delenv("ASFP8_INT8_DEQUANT", raising=False)
    else:
        monkeypatch.setenv("ASFP8_INT8_DEQUANT", env)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
    return patch._dequant_enabled()


def test_dequant_gate_is_opt_in(monkeypatch):
    """Unset stays OFF regardless of RAM."""
    assert _resolve_dequant_gate(monkeypatch, None, 128 * (1 << 30)) is False


def test_dequant_gate_off_stays_off(monkeypatch):
    assert _resolve_dequant_gate(monkeypatch, "off", 128 * (1 << 30)) is False


def test_dequant_gate_on_with_enough_ram(monkeypatch):
    assert _resolve_dequant_gate(monkeypatch, "1", 128 * (1 << 30)) is True


def test_dequant_gate_ram_check_overrides_opt_in(monkeypatch, capsys):
    """=1 on a small box is refused, and says so."""
    assert _resolve_dequant_gate(monkeypatch, "1", 16 * (1 << 30)) is False
    assert "48 GiB" in capsys.readouterr().out


def test_dequant_gate_memoises(monkeypatch):
    """The gate resolves once, so a later env change can't flip a live session."""
    assert _resolve_dequant_gate(monkeypatch, "1", 128 * (1 << 30)) is True
    monkeypatch.setenv("ASFP8_INT8_DEQUANT", "off")
    assert patch._dequant_enabled() is True


@requires_mps
def test_offloaded_cpu_weight_does_not_latch_the_kernel_off(monkeypatch):
    """An offloaded CPU weight doesn't latch the kernel off for the session (#26).

    Reaching _int8_linear_kernel with an MPS input and a CPU weight would throw in its
    own fallback and call mark_kernel_failed.
    """
    from comfy_kitchen.tensor import QuantizedTensor
    from _patches import _caps

    qw_cpu = QuantizedTensor.from_float(
        (torch.randn(64, 128) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    )
    assert qw_cpu.device.type == "cpu", "fixture must stay off-device"

    class Holder:
        weight = qw_cpu
        bias = None
        _full_precision_mm = False
        comfy_force_cast_weights = True
        weight_function = []
        bias_function = []

    x = torch.randn(8, 128, dtype=torch.bfloat16, device="mps")

    assert patch._try_int8_kernel_forward(Holder(), x) is None
    assert _caps._kernel_ready.get("int8") is not False, (
        "an offloaded weight latched the int8 kernel off for the session"
    )


@requires_mps
def test_dequant_leaves_an_offloaded_weight_on_the_cpu(monkeypatch):
    """An offloaded weight is left on the CPU, with the flag unset so the layer still
    dequantises once comfy brings it back."""
    from comfy_kitchen.tensor import QuantizedTensor

    monkeypatch.setattr(patch, "_DEQUANT_MODE", True, raising=False)

    qw_cpu = QuantizedTensor.from_float(
        (torch.randn(64, 128) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    )
    assert qw_cpu.device.type == "cpu", "fixture must stay offloaded"

    class Holder:
        _full_precision_mm = False
        comfy_force_cast_weights = True
        weight_function = []
        bias_function = []

    layer = Holder()
    layer.weight = qw_cpu
    layer.bias = None

    x = torch.randn(1, 128, dtype=torch.bfloat16, device="mps")
    patch._maybe_dequant_weight(layer, x)

    assert isinstance(layer.weight, QuantizedTensor), "offloaded weight was dequantised"
    assert layer.weight.device.type == "cpu", "offloaded weight was pulled onto the GPU"
    assert not getattr(layer, "_asfp8_deq_done", False), (
        "a transient offload latched the layer out of ever dequantising"
    )
