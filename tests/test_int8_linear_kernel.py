"""Tests for patch #17: INT8 W8A8 via the bit-exact Metal kernel (int8_linear_kernel_mps).

Unit tests (always run): install() is opt-in/guarded; the wrapper falls back to the
original int8_linear when the kernel is unavailable or off-MPS.

Integration test (opt-in, ASFP8_INT8_EXT=1 on an MPS box): the kernel-backed
int8_linear is bit-identical to comfy_kitchen's original across convrot/bias/3D/M=1.
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

# Run whenever the node ITSELF would use the kernel here -- same gate production
# uses. Hiding these behind an opt-in env var meant a kernel that stopped
# compiling (issue #13) showed up as 34 silent skips. ASFP8_INT8_EXT=0 still turns
# them off, exactly as it turns the feature off.
_int8_enabled = torch.backends.mps.is_available() and _caps.resolve(
    "ASFP8_INT8_EXT", default_on=True, cap=_caps.tier_b_ready
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
    """_caps.kernel_ready memoises per process.

    Without this, whichever test verifies the kernel first decides the answer for
    every later test, and a test that installs a deliberately broken kernel is
    silently skipped past its own gate.
    """
    from _patches import _caps
    _caps._kernel_ready.pop("int8", None)
    yield
    _caps._kernel_ready.pop("int8", None)


def test_install_noop_when_explicitly_off(monkeypatch):
    """ASFP8_INT8_EXT=off force-disables even on capable hardware."""
    from _patches import _caps
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: True)
    monkeypatch.setenv("ASFP8_INT8_EXT", "off")
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


def test_install_noop_when_not_capable(monkeypatch):
    """DEFAULT ON but gated: unsupported hardware -> no build attempt, stays inert."""
    from _patches import _caps
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: False)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


# --- the Metal build must never run on the ComfyUI startup thread ---------------
# Regression guard for "ComfyUI hangs forever at startup when `ninja` is installed":
# install() used to run a synchronous ninja+clang build of the ObjC++/Metal source
# while ComfyUI was still importing custom nodes, with no timeout and no message.


@requires_mps
def test_install_does_not_build_the_extension(monkeypatch):
    """install() runs at import time: it may wire the seam, never build the kernel."""
    from _patches import _caps
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: True)
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
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: True)
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
    """Direct call before install() (both _kernel and _orig are None) must raise a
    clear RuntimeError, not an opaque AttributeError from calling None(...)."""
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


# P0 verdict: Metal `erf` is unavailable under MTLLanguageVersion4_1, so act=3
# ("gelu", erf) is dropped entirely; only {silu, gelu_tanh} are supported.
@requires_int8_ext
@pytest.mark.parametrize("act", ["silu", "gelu_tanh"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("convrot", [False, True])  # rotation shifts the activation magnitude dist
@pytest.mark.parametrize("M", [1, 256])
def test_int8_linear_fused_activation_matches_reference(M, convrot, bias, act):
    """Fused-epilogue activation == torch activation of the unfused kernel output.

    convrot=True exercises the Hadamard-rotate -> requant path feeding the fused
    activation epilogue: rotation reshapes the magnitude distribution (GELU x^3,
    SiLU saturation, quant edges), so the activation must stay correct there too.
    """
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
    # The fused epilogue rounds the activation to bf16; torch rounds its own bf16
    # activation too, so the honest correctness bound is ONE bf16 ulp (relative
    # 2**-7 ~= 7.8e-3). The plan's rtol=2e-3 sits *below* bf16 precision and is
    # unsatisfiable for these magnitude-~15 outputs even by a perfect kernel
    # (the plan's rationale assumed outputs ~1.0; this data reaches ~15). rtol=8e-3
    # admits a 1-ulp-correct kernel; atol=2e-3 bounds the near-zero regime. This
    # still rejects real bugs: the original precise::tanh GELU (~160 ulp) had an
    # absolute diff of 0.0156 at small ref, which exceeds atol and fails here.
    # See docs/superpowers/results/D-results.md.
    d = (out.float() - ref.float()).abs()
    assert torch.allclose(out, ref, atol=2e-3, rtol=8e-3), \
        f"M={M} convrot={convrot} bias={bias} {act}: max|d|={d.max().item():.4g}"


def test_wrapper_fallback_applies_activation(monkeypatch):
    """Off-MPS / no-kernel fallback must still apply the requested activation, not drop it."""
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
    """Mirror of test_wrapper_rejects_unknown_act for the gated kernel: a typo'd
    activation must raise before any dispatch (CPU tensors, no kernel needed)."""
    w = torch.randint(-1, 2, (3, 4), dtype=torch.int8)
    s = torch.tensor([0.01])
    with pytest.raises(ValueError):
        patch._int8_swiglu_kernel(torch.randn(2, 4), w, w, s, s,
                                  None, None, False, 256, act="sillu")


def test_swiglu_rejects_none_act():
    """'none' is a *valid* activation name but meaningless for a gate — it must be
    rejected (the gate would degenerate to a plain elementwise product)."""
    w = torch.randint(-1, 2, (3, 4), dtype=torch.int8)
    s = torch.tensor([0.01])
    with pytest.raises(ValueError):
        patch._int8_swiglu_kernel(torch.randn(2, 4), w, w, s, s,
                                  None, None, False, 256, act="none")


# tolerance: one bf16 ulp (rtol=8e-3) + atol=2e-3 near zero — see the point-activation
# test rationale above and docs/superpowers/results/D-results.md. act=3 (gelu-erf) dropped.
# 2576 = remainder-K; the (256,2560,convrot=True) case exercises the Hadamard-rotate ->
# requant path through the fused gate kernel (rotation reshapes the gate magnitude dist,
# which is where SwiGLU/GEGLU saturation + quant edges live) and keeps its spy guard.
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
    # The fused gate keeps act(gate) in fp32 registers and multiplies by up in fp32,
    # rounding to bf16 ONCE (that single-rounding is the whole point of fusion). The
    # reference must do the same: torch's INDEPENDENT fp32 activation (not the kernel's
    # exp identity) * up in fp32, rounded once. A bf16-rounded intermediate gate would
    # double-round and diverge by up to half-a-gate-ulp * |up| — that is not a kernel
    # bug. The kernel matches this fp32-fused reference to ~1 bf16 ulp; near-zero gelu
    # epsilon * large up is bounded by atol=2e-3. See docs/superpowers/results/D-results.md.
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
    """Length-N weight scales must route through the per-branch path, not the fused gate kernel."""
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
    """The Metal library compiles on first kernel USE, not at extension build time.

    A toolchain that rejects it (issue #13) otherwise recompiles on every eligible
    Linear -- measured at 1.46x slower than not having the kernel at all, with 822
    fallback lines in one run. The cpp_extension build succeeds, so _kernel_tried
    never short-circuits this.
    """
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
    """Canary: if the node turns the int8 kernel on, it must actually work.

    The Metal library is compiled on first dispatch, so a macOS or toolchain
    update can kill it while the cpp_extension still builds and the capability
    banner stays green (issue #13). Without this the rest of the kernel tests
    just skip and the breakage is invisible.
    """
    assert patch._ensure_kernel() is not None, "int8 cpp_extension failed to build"
    assert patch._self_check(), (
        "int8 Metal library does not compile on this machine — the kernel is "
        "inert and every eligible Linear falls back to comfy's int8 path"
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_failed_verification_disables_the_kernel_and_is_not_retried(monkeypatch):
    """A kernel that fails verification must cost one attempt, not one per layer.

    This is the #14 gap: tier_b_ready() green-lights int8 off na_gemm's bf16
    probe, so a kernel that cannot build still reaches the forward path. The
    per-kernel memo is what stops that becoming a per-call rebuild.
    """
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
