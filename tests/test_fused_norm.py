# tests/test_fused_norm.py
"""Tests for the fused RMSNorm + affine + adaLN(scale,shift) + residual kernel.

The oracle is the module's own group-aware fp32 torch composition. Every MPS test asserts
_last_backend == "kernel" so none can pass through the fallback, and the set includes the
rows*D > 2**31 regime, since this kernel supersedes rmsnorm_mps_large.py.
"""
import pytest
import torch

from _patches import fused_norm_mps as m
from _patches.fused_norm_mps import fused_rmsnorm_modulate

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _reference(x, weight, eps, scale, shift, residual):
    # mirror the module's group-aware reference for tests that pre-expand inputs
    return m._reference(x, weight, eps, scale, shift, residual)


@mps
@pytest.mark.parametrize("dtype,atol,rtol", [
    (torch.float16, 5e-2, 5e-2),
    (torch.bfloat16, 8e-2, 8e-2),
    (torch.float32, 2e-4, 2e-4),
])
def test_full_path_matches_reference(dtype, atol, rtol):
    torch.manual_seed(0)
    rows, dim = 4096, 256
    x = torch.randn(rows, dim, device="mps", dtype=dtype)
    w = torch.randn(dim, device="mps", dtype=dtype)
    sc = torch.randn(dim, device="mps", dtype=dtype)
    sh = torch.randn(dim, device="mps", dtype=dtype)
    res = torch.randn(rows, dim, device="mps", dtype=dtype)
    ref = _reference(x, w, 1e-6, sc, sh, res)
    out = fused_rmsnorm_modulate(x, w, 1e-6, sc, sh, res)
    torch.mps.synchronize()
    assert m._last_backend == "kernel", "fell back to torch composition instead of running the kernel"
    assert out.dtype == dtype and out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)


@mps
def test_bare_rmsnorm_tight_tolerance():
    """A bare rmsnorm over x in [-1,1] with weight=ones: only final-store rounding, so the
    fp16 tolerance is tight."""
    torch.manual_seed(7)
    rows, dim = 1024, 64
    x = (torch.rand(rows, dim, device="mps", dtype=torch.float16) * 2.0 - 1.0)
    w = torch.ones(dim, device="mps", dtype=torch.float16)
    ref = _reference(x, w, 1e-6, None, None, None)
    out = fused_rmsnorm_modulate(x, w, 1e-6)
    torch.mps.synchronize()
    assert m._last_backend == "kernel"
    assert torch.allclose(out.float(), ref.float(), atol=3e-3, rtol=3e-3)


@mps
@pytest.mark.parametrize("use_scale,use_shift,use_res,use_w", [
    (False, False, False, True),    # bare rmsnorm + weight
    (True, False, False, True),     # + scale only
    (False, True, True, True),      # shift + residual, no scale
    (True, True, True, False),      # modulation + residual, no weight (weight=None)
])
def test_optional_args(use_scale, use_shift, use_res, use_w):
    torch.manual_seed(1)
    rows, dim = 2048, 320
    x = torch.randn(rows, dim, device="mps", dtype=torch.float16)
    w = torch.randn(dim, device="mps", dtype=torch.float16) if use_w else None
    sc = torch.randn(dim, device="mps", dtype=torch.float16) if use_scale else None
    sh = torch.randn(dim, device="mps", dtype=torch.float16) if use_shift else None
    res = torch.randn(rows, dim, device="mps", dtype=torch.float16) if use_res else None
    ref = _reference(x, w, 1e-6, sc, sh, res)
    out = fused_rmsnorm_modulate(x, w, 1e-6, sc, sh, res)
    torch.mps.synchronize()
    assert m._last_backend == "kernel"
    assert torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2)


@mps
def test_per_batch_modulation_grouping():
    """adaLN case: x flattened from [B, L, D], scale/shift are [B, D] (one group per batch)."""
    torch.manual_seed(2)
    B, L, D = 3, 512, 256
    x = torch.randn(B * L, D, device="mps", dtype=torch.float16)
    w = torch.randn(D, device="mps", dtype=torch.float16)
    sc = torch.randn(B, D, device="mps", dtype=torch.float16)
    sh = torch.randn(B, D, device="mps", dtype=torch.float16)
    res = torch.randn(B * L, D, device="mps", dtype=torch.float16)
    # group-aware reference expands [B,D] -> [B*L,D] exactly as the kernel maps row->group
    ref = _reference(x, w, 1e-6, sc, sh, res)
    out = fused_rmsnorm_modulate(x, w, 1e-6, sc, sh, res)
    torch.mps.synchronize()
    assert m._last_backend == "kernel"
    assert torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2)


@mps
def test_mixed_group_counts():
    """scale=[B,D] and shift=[1,D] together: different group counts in one call."""
    torch.manual_seed(8)
    B, L, D = 4, 256, 256
    x = torch.randn(B * L, D, device="mps", dtype=torch.float16)
    sc = torch.randn(B, D, device="mps", dtype=torch.float16)   # G=B
    sh = torch.randn(1, D, device="mps", dtype=torch.float16)   # G=1
    ref = _reference(x, None, 1e-6, sc, sh, None)
    out = fused_rmsnorm_modulate(x, None, 1e-6, sc, sh, None)
    torch.mps.synchronize()
    assert m._last_backend == "kernel"
    assert torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2)
    # reversed roles
    ref2 = _reference(x, None, 1e-6, sh, sc, None)
    out2 = fused_rmsnorm_modulate(x, None, 1e-6, sh, sc, None)
    torch.mps.synchronize()
    assert m._last_backend == "kernel"
    assert torch.allclose(out2.float(), ref2.float(), atol=5e-2, rtol=5e-2)


@mps
def test_multidim_normalized_shape_reroute():
    """The F.rms_norm reroute reduces over ALL normalized dims and reads a multi-dim weight."""
    m.install_for_test()                       # force the F.rms_norm reroute regardless of env flag
    try:
        torch.manual_seed(9)
        x = torch.randn(2, 3, 4, device="mps", dtype=torch.float16)
        w = torch.randn(3, 4, device="mps", dtype=torch.float16)
        # manual fp32 multi-dim rms_norm oracle (same formula as rmsnorm_mps_large.py)
        xf = x.float()
        dims = (1, 2)
        ref = (xf * torch.rsqrt(xf.pow(2).mean(dims, keepdim=True) + 1e-6)).to(x.dtype) * w
        out = torch.nn.functional.rms_norm(x, (3, 4), w, 1e-6)
        torch.mps.synchronize()
        assert m._last_backend == "kernel"
        assert out.shape == x.shape
        assert torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2)
    finally:
        m.uninstall_for_test()


@mps
def test_grouped_modulation_fallback_equiv(monkeypatch):
    """The fallback handles grouped [B,D] scale/shift, where plain broadcasting would crash."""
    torch.manual_seed(10)
    B, L, D = 3, 128, 256
    x = torch.randn(B * L, D, device="mps", dtype=torch.float16)
    sc = torch.randn(B, D, device="mps", dtype=torch.float16)
    sh = torch.randn(B, D, device="mps", dtype=torch.float16)
    ref = m._reference(x, None, 1e-6, sc, sh, None)        # group-aware oracle
    # force the kernel to fail so the wrapper takes the fallback
    monkeypatch.setattr(m, "_get_lib", lambda dtype: (_ for _ in ()).throw(RuntimeError("forced")))
    out = fused_rmsnorm_modulate(x, None, 1e-6, sc, sh, None)
    assert m._last_backend == "fallback", "expected the forced fallback path"
    assert torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2)


@mps
def test_bad_optional_shape_falls_back():
    """A short weight or mismatched residual routes to the fallback, never an OOB read."""
    torch.manual_seed(11)
    rows, dim = 256, 128
    x = torch.randn(rows, dim, device="mps", dtype=torch.float16)
    bad_w = torch.randn(dim - 1, device="mps", dtype=torch.float16)   # wrong length
    # the load-bearing assertion is the absence of a kernel dispatch: the torch fallback
    # then legitimately broadcast-fails on a genuinely malformed tensor
    m._last_backend = "kernel"
    with pytest.raises(RuntimeError):
        fused_rmsnorm_modulate(x, bad_w, 1e-6)
    assert m._last_backend == "fallback"
    bad_res = torch.randn(rows, dim + 1, device="mps", dtype=torch.float16)
    m._last_backend = "kernel"
    with pytest.raises(RuntimeError):
        fused_rmsnorm_modulate(x, None, 1e-6, None, None, bad_res)
    assert m._last_backend == "fallback"


@mps
@pytest.mark.slow
def test_overflow_rows_2pow24_kernel_path(monkeypatch):
    """rows*D past 2**31 goes through the real 64-bit-indexed kernel.

    _reference is poisoned so the fallback cannot answer instead. Needs ~8.6 GiB.
    """
    monkeypatch.setattr(m, "_reference",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fall back")))
    torch.manual_seed(4)
    rows, dim = 1 << 24, 256
    x = torch.randn(rows, dim, device="mps", dtype=torch.float16)
    w = torch.randn(dim, device="mps", dtype=torch.float16)
    out = fused_rmsnorm_modulate(x, w, 1e-6)
    torch.mps.synchronize()
    assert m._last_backend == "kernel"
    idx = torch.tensor([0, rows // 3, rows - 1], device="mps")
    xr = x.index_select(0, idx).float()
    ref = (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()
    assert torch.isfinite(out.index_select(0, idx).float()).all()
    assert torch.allclose(out.index_select(0, idx).float(), ref.float(), atol=5e-2, rtol=5e-2)


@mps
@pytest.mark.slow
def test_stock_mps_broken_regime_2pow22():
    """The kernel stays correct at 1<<22 rows, the regime where stock MPS rms_norm breaks.

    Below the int32 element max, so this is the row-count bug, not offset overflow.
    """
    torch.manual_seed(3)
    rows, dim = 1 << 22, 256
    x = torch.randn(rows, dim, device="mps", dtype=torch.float16)
    w = torch.randn(dim, device="mps", dtype=torch.float16)
    out = fused_rmsnorm_modulate(x, w, 1e-6)
    torch.mps.synchronize()
    assert m._last_backend == "kernel"
    idx = torch.tensor([0, 1, rows // 2, rows - 2, rows - 1], device="mps")
    xr = x.index_select(0, idx).float()
    ref = (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()
    assert torch.isfinite(out.index_select(0, idx).float()).all()
    assert torch.allclose(out.index_select(0, idx).float(), ref.float(), atol=5e-2, rtol=5e-2)


def test_cpu_falls_back():
    """Off-MPS input must hit the torch-composition fallback, not the kernel."""
    x = torch.randn(8, 16)            # CPU tensor
    w = torch.randn(16)
    out = m.fused_rmsnorm_modulate(x, w, 1e-6)
    ref = m._reference(x, w, 1e-6, None, None, None)
    assert m._last_backend == "fallback"
    assert torch.allclose(out, ref, atol=1e-6)


@mps
@pytest.mark.parametrize("D", [17, 31])
def test_direct_kernel_D_not_multiple_of_32(D):
    """A D that isn't a multiple of 32 reduces correctly, against an independent fp32 oracle."""
    torch.manual_seed(D)
    rows = 257                                            # not a multiple of TG either
    x = torch.randn(rows, D, device="mps", dtype=torch.float16)
    w = torch.randn(D, device="mps", dtype=torch.float16)
    out = fused_rmsnorm_modulate(x, w, 1e-6)
    torch.mps.synchronize()
    assert m._last_backend == "kernel"
    # hand-computed fp32 oracle: rmsnorm(x) * weight, eps inside sqrt (LLaMA formulation)
    xf = x.float()
    oracle = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()
    assert torch.allclose(out.float(), oracle, atol=5e-2, rtol=5e-2)


def test_reroute_bypass_sets_fallback_backend():
    """An outer-guard bypass sets _last_backend = "fallback", so no stale "kernel" survives."""
    m.install_for_test()
    try:
        m._last_backend = "kernel"                        # a stale spy from a prior run
        x = torch.randn(4, 16)                            # CPU -> the outer bypass
        w = torch.randn(16)
        out = torch.nn.functional.rms_norm(x, (16,), w, 1e-6)
        assert m._last_backend == "fallback", "outer reroute bypass must reset the spy to 'fallback'"
        assert out.shape == x.shape
    finally:
        m.uninstall_for_test()
