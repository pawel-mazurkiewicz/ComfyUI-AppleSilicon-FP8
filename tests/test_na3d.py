"""Tests for the fused neighborhood attention (comfy_kitchen.na3d) kernel on MPS, against
comfy_kitchen's eager implementation. Ported from Illuminus7/ComfyUI-AppleSilicon-int8-Fast (MIT)."""
import sys

import pytest
import torch

from _patches import _caps
from _patches import na3d_mps as N

DTYPES = [torch.bfloat16, torch.float16]


@pytest.fixture(scope="module")
def ref_na3d():
    eager = pytest.importorskip("comfy_kitchen.backends.eager.na")
    if not _caps.shader_gate():
        pytest.skip("needs MPS compile_shader on an M5-class GPU")
    return eager.na3d


@pytest.fixture
def calls(ref_na3d, monkeypatch):
    """Stand-in for the original na3d: records each call it receives."""
    seen = []
    monkeypatch.setattr(N, "_orig", lambda *a: seen.append(a) or a[0])
    return seen


def _err(a, ref):
    a, ref = a.cpu().float(), ref.cpu().float()
    return ((a - ref).norm() / ref.norm()).item()


def _rand(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=g).to(dtype) for _ in range(3)]


# --- kernel numerics -------------------------------------------------------------


def test_shader_verifies(ref_na3d):
    assert N._verify()
    assert N._good == {torch.bfloat16, torch.float16}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,ks", [
    ((1, 9, 13, 10, 3, 64), (3, 7, 7)),       # ragged bricks on every axis
    ((2, 4, 6, 5, 2, 64), (11, 11, 11)),      # kernel clamped to every axis
    ((1, 12, 20, 18, 1, 64), (11, 11, 11)),   # interior windows, union larger than one key block
])
def test_matches_eager(ref_na3d, calls, shape, ks, dtype):
    q, k, v = _rand(shape, dtype, 1)
    ref = ref_na3d(q.float(), k.float(), v.float(), list(ks), None, 0.125)
    out = N._na3d(q.to("mps"), k.to("mps"), v.to("mps"), list(ks), None, 0.125)
    assert not calls
    assert out.shape == shape and out.dtype == dtype and _err(out, ref) < 0.01


@pytest.mark.parametrize("dtype", DTYPES)
def test_peaked_softmax_at_scale_one(ref_na3d, calls, dtype):
    """The LTX decoder passes pre-scaled queries and scale=1.0, so scores can be large."""
    q, k, v = _rand((1, 6, 10, 10, 2, 64), dtype, 3)
    q = (q.float() * 4).to(dtype)
    ref = ref_na3d(q.float(), k.float(), v.float(), [3, 5, 5], None, 1.0)
    out = N._na3d(q.to("mps"), k.to("mps"), v.to("mps"), [3, 5, 5], None, 1.0)
    assert not calls
    assert _err(out, ref) < 0.01


@pytest.mark.parametrize("dtype", DTYPES)
def test_qkv_views_of_one_buffer(ref_na3d, calls, dtype):
    """q/k/v as strided views of a fused qkv buffer; output contiguous; scale=None is d**-0.5."""
    g = torch.Generator().manual_seed(2)
    qkv = torch.randn(1, 5, 9, 11, 3, 2, 64, generator=g).to(dtype)
    q, k, v = qkv.unbind(4)
    ref = ref_na3d(q.float(), k.float(), v.float(), [3, 5, 5], None, 0.125)
    out = N._na3d(*qkv.to("mps").unbind(4), [3, 5, 5], None, None)
    assert not calls
    assert out.is_contiguous() and _err(out, ref) < 0.01


def test_union_at_the_uchar_limit(ref_na3d, calls):
    """Window 252 + 3 brick shift = a 255-wide union, the widest the byte-packed coords allow."""
    q, k, v = _rand((1, 300, 4, 4, 1, 64), torch.bfloat16, 4)
    ref = ref_na3d(q.float(), k.float(), v.float(), [252, 3, 3], None, 0.125)
    out = N._na3d(q.to("mps"), k.to("mps"), v.to("mps"), [252, 3, 3], None, 0.125)
    assert not calls
    assert _err(out, ref) < 0.01


# --- the byte-packing guard (runs anywhere) --------------------------------------


@pytest.mark.parametrize("ks,dims,want", [
    ([3, 5, 5], (10, 10, 10), [3, 5, 5]),
    ([11, 11, 11], (4, 6, 5), [4, 6, 5]),          # clamped to the axis
    (7, (10, 10, 10), [7, 7, 7]),                  # bare int repeats
    ([252, 3, 3], (300, 10, 10), [252, 3, 3]),     # union 255: fits
    ([253, 3, 3], (300, 10, 10), None),            # window fits, union 256 doesn't
    ([256, 3, 3], (300, 10, 10), None),            # window itself too wide
    ([3, 3, 300], (10, 10, 600), None),
    ([253, 3, 3], (254, 10, 10), [253, 3, 3]),     # union capped by a 254-long axis
    ([300, 3, 3], (200, 10, 10), [200, 3, 3]),     # clamped below the limit
    ([3, 3], (10, 10, 10), None),                  # wrong arity
    ([0, 3, 3], (10, 10, 10), None),
    ([3.0, 3, 3], (10, 10, 10), None),
])
def test_clamped_kernels_guard(ks, dims, want):
    assert N._clamped_kernels(ks, dims) == want


def test_extent_over_255_takes_the_original(ref_na3d, calls):
    x = torch.randn(1, 300, 4, 4, 1, 64, device="mps", dtype=torch.bfloat16)
    N._na3d(x, x, x, [253, 3, 3], None, 0.125)
    assert len(calls) == 1


# --- routing ---------------------------------------------------------------------


def test_unsupported_calls_take_the_original(ref_na3d, calls):
    x = torch.randn(1, 3, 4, 4, 2, 64, device="mps", dtype=torch.bfloat16)
    N._na3d(x, x, x, [3, 3, 3], [True, False, False], None)                # causal
    N._na3d(x, x, x, [3, 3, 3], True, None)                                 # causal, bare bool
    y = torch.randn(1, 3, 4, 4, 2, 32, device="mps", dtype=torch.bfloat16)
    N._na3d(y, y, y, [3, 3, 3], None, None)                                 # head_dim 32
    N._na3d(x.float(), x.float(), x.float(), [3, 3, 3], None, None)         # fp32
    N._na3d(x.cpu(), x.cpu(), x.cpu(), [3, 3, 3], None, None)               # cpu
    N._na3d(x, x, x, [3, 3], None, None)                                    # wrong arity
    N._na3d(x, x, x, [3, 3, 3], None, torch.tensor(0.125))                  # tensor scale
    assert len(calls) == 7


def test_grad_takes_the_original_only_when_needed(ref_na3d, calls):
    x = torch.randn(1, 3, 4, 4, 2, 64, device="mps", dtype=torch.bfloat16)
    xg = x.clone().requires_grad_()
    N._na3d(xg, x, x, [3, 3, 3], None, None)
    assert len(calls) == 1
    with torch.no_grad():
        N._na3d(xg, x, x, [3, 3, 3], None, None)
    assert len(calls) == 1


def test_mismatched_or_misaligned_layouts_take_the_original(ref_na3d, calls):
    shape = (1, 3, 4, 4, 2, 64)
    x = torch.randn(shape, device="mps", dtype=torch.bfloat16)
    qkv = torch.randn(1, 3, 4, 4, 3, 2, 64, device="mps", dtype=torch.bfloat16)
    N._na3d(x, qkv.unbind(4)[1], qkv.unbind(4)[2], [3, 3, 3], None, None)   # q strides differ from k/v
    buf = torch.randn(1 + x.numel(), device="mps", dtype=torch.bfloat16)
    off = buf[1:].view(shape)                                               # storage offset 1
    N._na3d(off, off, off, [3, 3, 3], None, None)
    assert len(calls) == 2


def test_kernel_failure_latches_off(ref_na3d, calls, monkeypatch):
    assert _caps.kernel_ready("na3d", N._verify)
    monkeypatch.setitem(_caps._kernel_ready, "na3d", True)   # restored afterwards
    launches = []

    def boom(*a, **k):
        launches.append(1)
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(N, "_launch", boom)
    x = torch.randn(1, 3, 4, 4, 2, 64, device="mps", dtype=torch.bfloat16)
    N._na3d(x, x, x, [3, 3, 3], None, None)
    N._na3d(x, x, x, [3, 3, 3], None, None)
    assert launches == [1] and len(calls) == 2
    assert _caps._kernel_ready["na3d"] is False


# --- install (runs anywhere on macOS) --------------------------------------------


@pytest.fixture
def ck(monkeypatch):
    mod = pytest.importorskip("comfy_kitchen")
    monkeypatch.setattr(N, "_installed", False)
    monkeypatch.setattr(N, "_orig", None)
    monkeypatch.delenv("ASFP8_NA3D", raising=False)
    return mod


needs_darwin = pytest.mark.skipif(sys.platform != "darwin", reason="install() is macOS-only")


@needs_darwin
def test_install_is_inert_off_mps(ck, monkeypatch):
    before = ck.na3d
    monkeypatch.setattr(_caps, "is_mps", lambda: False)
    monkeypatch.setenv("ASFP8_NA3D", "on")   # even forced on
    N.install()
    assert ck.na3d is before and not N._installed


@needs_darwin
def test_install_respects_off_and_the_capability_gate(ck, monkeypatch):
    before = ck.na3d
    monkeypatch.setattr(_caps, "is_mps", lambda: True)
    monkeypatch.setattr(_caps, "shader_gate", lambda: False)
    N.install()
    assert ck.na3d is before
    monkeypatch.setattr(_caps, "shader_gate", lambda: True)
    monkeypatch.setenv("ASFP8_NA3D", "off")
    N.install()
    assert ck.na3d is before and not N._installed


@needs_darwin
def test_install_is_inert_without_comfy_kitchen_na3d(ck, monkeypatch):
    monkeypatch.setattr(_caps, "is_mps", lambda: True)
    monkeypatch.setattr(_caps, "shader_gate", lambda: True)
    monkeypatch.delattr(ck, "na3d")
    N.install()
    assert not hasattr(ck, "na3d") and not N._installed


@needs_darwin
def test_install_patches_only_the_module_attribute_and_is_idempotent(ck, monkeypatch):
    monkeypatch.setattr(_caps, "is_mps", lambda: True)
    monkeypatch.setattr(_caps, "shader_gate", lambda: True)
    original = lambda *a: a[0]   # noqa: E731
    monkeypatch.setattr(ck, "na3d", original)
    N.install()
    assert ck.na3d is N._na3d and N._orig is original and N._installed
    N.install()
    N._installed = False   # a re-import must not wrap the wrapper
    N.install()
    assert ck.na3d is N._na3d and N._orig is original
