"""Wan VAE channel RMS_norm kernel (wan_vae_norm_mps) against the stock RMS_norm.forward.

Every kernel-path test counts _channel_norm calls so none can pass through the fallback;
every fallback test asserts the kernel was NOT called and the output equals stock exactly.
"""
import os
import sys
import types

import pytest
import torch

# ComfyUI-desktop keeps the code tree outside the repo venv
for _c in (os.environ.get("ASFP8_COMFY_PATH"), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
           os.path.expanduser("~/ComfyUI-Installs/ComfyUI/ComfyUI")):
    if _c and os.path.isdir(os.path.join(_c, "comfy")) and _c not in sys.path:
        sys.path.insert(0, _c)

vae = pytest.importorskip("comfy.ldm.wan.vae")

from _patches import _caps  # noqa: E402
from _patches import wan_vae_norm_mps as m  # noqa: E402

RMS_norm = vae.RMS_norm
STOCK = RMS_norm.forward   # captured before any install

gpu = pytest.mark.skipif(
    not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")),
    reason="needs MPS + torch.mps.compile_shader",
)

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# max relative L2 error vs the stock forward run in the same dtype (stock rounds after every op)
TOL = {torch.bfloat16: 1.5e-2, torch.float16: 3e-3, torch.float32: 1e-5}


def _norm(c, *, images=False, bias=True, channel_first=True, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = RMS_norm(c, channel_first=channel_first, images=images, bias=bias)
    with torch.no_grad():
        n.gamma.copy_(torch.randn(n.gamma.shape, generator=g))
        if bias:
            n.bias.copy_(torch.randn(n.bias.shape, generator=g))
    return n.to("mps", dtype)


def _x(shape, dtype, seed=1, device="mps"):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(device, dtype)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.fixture
def clean(monkeypatch):
    """Pristine module/class state; monkeypatch restores it afterwards."""
    monkeypatch.setattr(RMS_norm, "forward", STOCK)
    monkeypatch.setattr(m, "_installed", False)
    monkeypatch.setattr(m, "_orig_forward", None)
    monkeypatch.setattr(m, "_libs", {})
    monkeypatch.setenv("ASFP8_WAN_VAE_NORM", "on")
    _caps._kernel_ready.pop(m.NAME, None)
    yield monkeypatch
    _caps._kernel_ready.pop(m.NAME, None)


@pytest.fixture
def installed(clean):
    m.install()
    assert RMS_norm.forward is m._forward
    return clean


@pytest.fixture
def patched(installed):
    """Installed and self-checked, so later spies count only real forwards."""
    assert _caps.kernel_ready(m.NAME, m._verify), "kernel failed its own self-check"
    assert set(m._libs) == set(DTYPES)
    return installed


@pytest.fixture
def calls(patched):
    real = m._channel_norm
    n = []

    def spy(*a, **k):
        n.append(1)
        return real(*a, **k)

    patched.setattr(m, "_channel_norm", spy)
    return n


def _run(n, x):
    with torch.no_grad():
        return STOCK(n, x), n(x)


# ---- kernel path matches stock ------------------------------------------------------------

@gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("shape", [(2, 24, 3, 6, 10), (1, 96, 2, 16, 24), (1, 384, 1, 12, 20)])
def test_5d_video_matches_stock(calls, dtype, bias, shape):
    n = _norm(shape[1], bias=bias, dtype=dtype)
    ref, out = _run(n, _x(shape, dtype))
    assert len(calls) == 1, "fell back instead of running the kernel"
    assert out.dtype == dtype and out.shape == ref.shape
    assert _rel(out, ref) < TOL[dtype]


@gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("shape", [(2, 32, 8, 12), (1, 96, 16, 16)])
def test_4d_image_matches_stock(calls, dtype, bias, shape):
    n = _norm(shape[1], images=True, bias=bias, dtype=dtype)
    ref, out = _run(n, _x(shape, dtype))
    assert len(calls) == 1
    assert out.dtype == dtype and out.shape == ref.shape
    assert _rel(out, ref) < TOL[dtype]


@gpu
def test_fp32_params_with_bf16_input(calls):
    n = _norm(24, dtype=torch.float32)
    ref, out = _run(n, _x((1, 24, 2, 6, 10), torch.bfloat16))
    assert len(calls) == 1
    assert _rel(out, ref) < TOL[torch.bfloat16]


@gpu
def test_at_least_as_accurate_as_stock(calls):
    n = _norm(96, dtype=torch.bfloat16)
    x = _x((1, 96, 2, 16, 24), torch.bfloat16)
    with torch.no_grad():
        oracle = STOCK(n.to("cpu", torch.float32), x.cpu().float())
        n.to("mps", torch.bfloat16)
        ref, out = STOCK(n, x), n(x)
    assert len(calls) == 1
    assert _rel(out.cpu(), oracle) <= _rel(ref.cpu(), oracle) * 1.05 + 1e-4


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_zero_positions_stay_finite_and_match(calls, dtype):
    x = _x((1, 24, 2, 6, 10), dtype)
    x[:, :, :, :, :4] = 0
    n = _norm(24, dtype=dtype)
    ref, out = _run(n, x)
    assert len(calls) == 1
    assert torch.isfinite(out.float()).all()
    assert _rel(out, ref) < TOL[dtype]


@gpu
def test_non_contiguous_input_uses_kernel(calls):
    n = _norm(24, dtype=torch.float32)
    x = _x((1, 24, 4, 6, 10), torch.float32)[:, :, 1:3]
    assert not x.is_contiguous()
    ref, out = _run(n, x)
    assert len(calls) == 1
    assert _rel(out, ref) < TOL[torch.float32]


# ---- fp32 affine cache --------------------------------------------------------------------

@gpu
def test_bf16_params_cast_once_then_refreshed_on_change(calls):
    n = _norm(24, dtype=torch.bfloat16)
    x = _x((1, 24, 2, 6, 10), torch.bfloat16)
    _run(n, x)
    g0, b0 = n.__dict__["_asfp8_wan_norm"][1:]
    assert g0.dtype == b0.dtype == torch.float32
    _run(n, x)
    g1, b1 = n.__dict__["_asfp8_wan_norm"][1:]
    assert g1 is g0 and b1 is b0, "re-cast the affine params on a repeat call"
    with torch.no_grad():
        n.gamma.mul_(2)   # in place: bumps the version counter
    ref, out = _run(n, x)
    assert n.__dict__["_asfp8_wan_norm"][1] is not g0
    assert _rel(out, ref) < TOL[torch.bfloat16]


@gpu
def test_fp32_params_are_not_copied(calls):
    n = _norm(24, dtype=torch.float32)
    _run(n, _x((1, 24, 2, 6, 10), torch.float32))
    g = n.__dict__["_asfp8_wan_norm"][1]
    assert g.data_ptr() == n.gamma.data_ptr()


def test_affine_rejects_non_stock_broadcast_shapes():
    x5 = torch.zeros(1, 24, 2, 4, 4)
    assert m._affine(RMS_norm(24, images=True), x5) is None       # gamma (C,1,1) vs 5-D x
    assert m._affine(RMS_norm(24, images=False), torch.zeros(1, 24, 4, 4)) is None
    assert m._affine(RMS_norm(24, images=False), x5) is not None


# ---- fallbacks: kernel not called, result is stock's ----------------------------------------

def _assert_stock(calls, n, x):
    ref, out = _run(n, x)
    assert calls == [], "kernel ran where a fallback was required"
    assert torch.equal(out, ref)


@gpu
def test_fallback_spatial_not_multiple_of_4(calls):
    _assert_stock(calls, _norm(24), _x((1, 24, 2, 5, 7), torch.float32))


@gpu
def test_fallback_cpu_tensors(calls):
    n = _norm(24).to("cpu")
    _assert_stock(calls, n, _x((1, 24, 2, 6, 10), torch.float32, device="cpu"))


@gpu
def test_fallback_channel_last_norm(calls):
    n = _norm(24, images=False, channel_first=False)
    _assert_stock(calls, n, _x((2, 5, 8, 24), torch.float32))


@gpu
def test_fallback_unaligned_storage_offset(calls):
    n = _norm(24)
    flat = _x((1 + 24 * 2 * 6 * 10,), torch.float32)
    x = flat[1:].view(1, 24, 2, 6, 10)
    assert x.storage_offset() == 1
    _assert_stock(calls, n, x)


@gpu
def test_fallback_empty_input(calls):
    _assert_stock(calls, _norm(24), torch.zeros(0, 24, 1, 4, 4, device="mps"))


@gpu
def test_fallback_when_grad_needed_for_input(calls):
    n = _norm(24).requires_grad_(False)
    x = _x((1, 24, 2, 6, 10), torch.float32).requires_grad_(True)
    ref, out = STOCK(n, x), n(x)
    assert calls == []
    assert out.requires_grad and torch.equal(out, ref)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@gpu
def test_fallback_when_grad_needed_for_params(calls):
    n = _norm(24)   # params require grad by default
    x = _x((1, 24, 2, 6, 10), torch.float32)
    assert torch.is_grad_enabled()
    out = n(x)
    assert calls == [] and out.requires_grad
    out.sum().backward()
    assert n.gamma.grad is not None


@gpu
@pytest.mark.parametrize("ctx", [torch.no_grad, torch.inference_mode], ids=["no_grad", "inference_mode"])
def test_kernel_used_without_autograd_even_if_params_require_grad(calls, ctx):
    n = _norm(24)
    x = _x((1, 24, 2, 6, 10), torch.float32).requires_grad_(True)
    with ctx():
        out = n(x)
        ref = STOCK(n, x)
    assert len(calls) == 1 and not out.requires_grad
    assert _rel(out, ref) < TOL[torch.float32]


@gpu
def test_kernel_used_when_nothing_requires_grad(calls):
    n = _norm(24).requires_grad_(False)
    x = _x((1, 24, 2, 6, 10), torch.float32)
    assert torch.is_grad_enabled()
    n(x)
    assert len(calls) == 1


@gpu
def test_inference_mode_params_skip_cache_but_still_work(calls):
    with torch.inference_mode():
        n = _norm(24)   # params are inference tensors: no version counter
    x = _x((1, 24, 2, 6, 10), torch.float32)
    with torch.inference_mode():
        out, ref = n(x), STOCK(n, x)
    assert len(calls) == 1 and "_asfp8_wan_norm" not in n.__dict__
    assert _rel(out, ref) < TOL[torch.float32]


# ---- latching, verification, install ---------------------------------------------------------

@gpu
def test_kernel_exception_falls_back_and_latches_off(patched):
    seen = []

    def boom(*a, **k):
        seen.append(1)
        raise RuntimeError("metal dispatch failed")

    patched.setattr(m, "_channel_norm", boom)
    n = _norm(24)
    x = _x((1, 24, 2, 6, 10), torch.float32)
    ref, out = _run(n, x)
    assert seen == [1] and torch.equal(out, ref)
    assert _caps.kernel_ready(m.NAME, lambda: True) is False
    out2 = _run(n, x)[1]
    assert seen == [1], "kernel retried after latching off"
    assert torch.equal(out2, ref)


@gpu
def test_failed_self_check_means_stock_forever(installed):
    installed.setattr(m, "_verify", lambda: False)
    n = _norm(24)
    x = _x((1, 24, 2, 6, 10), torch.float32)
    boom = []
    installed.setattr(m, "_channel_norm", lambda *a, **k: boom.append(1))
    ref, out = _run(n, x)
    assert boom == [] and torch.equal(out, ref)


@gpu
def test_verify_rejects_a_numerically_wrong_kernel(clean):
    broken = m._MSL.replace("* r * gamma[c]", "* r * 2.0f * gamma[c]")
    assert broken != m._MSL
    clean.setattr(m, "_MSL", broken)
    assert m._verify() is False
    assert m._libs == {}


@gpu
def test_verify_passes_each_dtype(clean):
    assert m._verify() is True
    assert set(m._libs) == set(DTYPES)


@gpu
def test_install_is_idempotent(patched):
    m.install()
    m.install()
    assert RMS_norm.forward is m._forward
    assert m._orig_forward is STOCK


def test_install_respects_env_off(clean):
    clean.setenv("ASFP8_WAN_VAE_NORM", "off")
    m.install()
    assert RMS_norm.forward is STOCK and not m._installed


def test_install_inert_without_compile_shader(clean):
    clean.setattr(_caps, "has_compile_shader", lambda: False)
    m.install()
    assert RMS_norm.forward is STOCK and not m._installed and m._orig_forward is None


def test_install_inert_when_class_missing(clean):
    clean.setattr(_caps, "has_compile_shader", lambda: True)
    clean.setitem(sys.modules, "comfy.ldm.wan.vae", types.ModuleType("comfy.ldm.wan.vae"))
    m.install()
    assert not m._installed and m._orig_forward is None
    assert RMS_norm.forward is STOCK


def test_install_inert_when_forward_missing(clean):
    class Bare:
        pass

    stub = types.ModuleType("comfy.ldm.wan.vae")
    stub.RMS_norm = Bare
    clean.setattr(_caps, "has_compile_shader", lambda: True)
    clean.setitem(sys.modules, "comfy.ldm.wan.vae", stub)
    m.install()
    assert not m._installed and m._orig_forward is None
    assert not hasattr(Bare, "forward")
