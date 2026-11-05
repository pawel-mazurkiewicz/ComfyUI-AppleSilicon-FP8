"""The one-slot INT8 activation cache must be exact, identity-keyed, and hold no strong ref."""
import gc
import weakref

import pytest
import torch

from conftest import requires_mps

from _patches import int8_linear_kernel_mps as patch

ENV = "ASFP8_INT8_ACT_CACHE"


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    patch._ACT_CACHE = None
    yield
    patch._ACT_CACHE = None


def _x():
    return torch.randn(64, 512, dtype=torch.bfloat16, device="mps")


def _spy(monkeypatch):
    # the ptr, not the tensor: a strong ref here would defeat test_cache_holds_no_strong_ref
    real = patch.int8_quant_mps.quantize_rowwise
    calls = []

    def spy(x, *a, **k):
        calls.append(x.data_ptr())
        return real(x, *a, **k)

    monkeypatch.setattr(patch.int8_quant_mps, "quantize_rowwise", spy)
    return calls


@requires_mps
def test_cache_reuses_for_same_tensor(monkeypatch):
    calls = _spy(monkeypatch)
    x = _x()
    got = [patch._rotate_quant(x, False, 256) for _ in range(3)]
    assert len(calls) == 1
    monkeypatch.setenv(ENV, "off")
    q, s = patch._rotate_quant(x, False, 256)
    assert all(torch.equal(a, q) and torch.equal(b, s) for a, b in got)


@requires_mps
def test_cache_misses_on_new_tensor(monkeypatch):
    calls = _spy(monkeypatch)
    x = _x()
    patch._rotate_quant(x, False, 256)
    patch._rotate_quant(x.clone(), False, 256)
    assert len(calls) == 2


@requires_mps
def test_cache_misses_on_inplace_edit(monkeypatch):
    calls = _spy(monkeypatch)
    x = _x()
    assert not x.is_inference()
    patch._rotate_quant(x, False, 256)
    x.add_(1)
    got = patch._rotate_quant(x, False, 256)
    assert len(calls) == 2
    monkeypatch.setenv(ENV, "off")
    q, s = patch._rotate_quant(x, False, 256)
    assert torch.equal(got[0], q) and torch.equal(got[1], s)


@requires_mps
def test_cache_misses_on_convrot_change(monkeypatch):
    calls = _spy(monkeypatch)
    x = _x()
    patch._rotate_quant(x, False, 256)
    got = patch._rotate_quant(x, True, 256)
    assert len(calls) == 2
    monkeypatch.setenv(ENV, "off")
    q, s = patch._rotate_quant(x, True, 256)
    assert torch.equal(got[0], q) and torch.equal(got[1], s)


@requires_mps
def test_cache_off_switch(monkeypatch):
    calls = _spy(monkeypatch)
    monkeypatch.setenv(ENV, "off")
    x = _x()
    for _ in range(3):
        patch._rotate_quant(x, False, 256)
    assert len(calls) == 3


@requires_mps
def test_cache_holds_no_strong_ref(monkeypatch):
    x = _x()
    r = weakref.ref(x)
    patch._rotate_quant(x, False, 256)
    del x
    gc.collect()
    assert r() is None


@requires_mps
def test_inference_tensor_hits_by_identity(monkeypatch):
    calls = _spy(monkeypatch)
    with torch.inference_mode():
        x = torch.randn(64, 512, dtype=torch.bfloat16, device="mps")
        assert x.is_inference()
        a = patch._rotate_quant(x, False, 256)
        b = patch._rotate_quant(x, False, 256)
    assert len(calls) == 1
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@requires_mps
def test_linear_kernel_uses_cache(monkeypatch):
    calls = _spy(monkeypatch)

    class FakeKernel:
        def i8_matmul2d_nt_fused_cs(self, x8, w, xs, ws, bias, act):
            return torch.zeros(x8.shape[0], w.shape[0], dtype=torch.bfloat16, device="mps")

    monkeypatch.setattr(patch, "_kernel", FakeKernel())
    monkeypatch.setattr(patch, "_orig_int8_linear",
                        lambda *a, **k: pytest.fail("fell back to comfy's int8_linear"))
    x = _x()
    for n in (16, 32, 24):
        w = torch.randint(-128, 128, (n, 512), dtype=torch.int8, device="mps")
        ws = torch.rand(n, device="mps")
        out = patch._int8_linear_kernel(x, w, ws, None, torch.bfloat16, False, 256)
        assert out.shape == (64, n)
    assert len(calls) == 1


@requires_mps
def test_swiglu_kernel_uses_cache(monkeypatch):
    calls = _spy(monkeypatch)

    class FakeKernel:
        def i8_matmul2d_nt_swiglu_cs(self, x8, wg, wu, xs, wsg, wsu, bg, bu, act):
            return torch.zeros(x8.shape[0], wg.shape[0], dtype=torch.bfloat16, device="mps")

    monkeypatch.setattr(patch, "_kernel", FakeKernel())
    x = _x()
    for n in (16, 32):
        wg = torch.randint(-128, 128, (n, 512), dtype=torch.int8, device="mps")
        wu = torch.randint(-128, 128, (n, 512), dtype=torch.int8, device="mps")
        wsg, wsu = torch.rand(n, device="mps"), torch.rand(n, device="mps")
        out = patch._int8_swiglu_kernel(x, wg, wu, wsg, wsu)
        assert out.shape == (64, n)
    assert len(calls) == 1
