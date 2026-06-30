# tests/test_fused_norm.py
"""Issue E: fused RMSNorm + affine + adaLN(scale,shift) + residual kernel.

Correctness oracle = exact, GROUP-AWARE torch composition in fp32 (the same formula the
wrapper falls back to). Every MPS test asserts the real Metal kernel ran (m._last_backend
== 'kernel') so it cannot pass silently through the torch fallback. Includes the int32
row*D overflow regime (rows*D > 2**31) since this kernel supersedes rmsnorm_mps_large.py."""
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
    """Deterministic low-dynamic-range case: x in [-1,1], weight=ones, no modulation/residual.
    Expected error is final-store rounding only, so use a tight fp16 tolerance (Codex MINOR #10)."""
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
    """Open question #5/#6: scale=[B,D], shift=[1,D] (different group counts) must work."""
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
    """Codex BLOCKER #2: F.rms_norm reroute must reduce over ALL normalized dims
    (D = prod(normalized_shape)), not just the last one, and must read a multi-dim weight."""
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
    """Codex MAJOR #3: forcing the fallback with grouped [B,D] scale/shift must NOT raise and
    must match the kernel result (group-aware reference). Plain broadcasting would crash here."""
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
    """Codex MAJOR #4: a short weight / mismatched residual must route to the fallback, not
    dispatch undefined memory reads."""
    torch.manual_seed(11)
    rows, dim = 256, 128
    x = torch.randn(rows, dim, device="mps", dtype=torch.float16)
    bad_w = torch.randn(dim - 1, device="mps", dtype=torch.float16)   # wrong length
    # the validation must route to the fallback (no kernel dispatch / no OOB buffer read); the
    # torch fallback then legitimately broadcast-fails on the genuinely-malformed tensor. The
    # load-bearing assertion is that we did NOT dispatch the kernel (backend == "fallback").
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
    """THE int32-offset overflow proof (Codex MAJOR #5): rows*D = 1<<24 * 256 = 4.29e9 > 2**31.
    Monkeypatch _reference to raise so this CANNOT pass through the fallback — it must be the
    real 64-bit-indexed kernel. ~8.6 GiB fp16; needs the 128 GB box."""
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
    """rows=1<<22 (4.19M). This is BELOW int32 element max (1.07e9 < 2.147e9) so it does NOT
    prove offset overflow; it reproduces the *stock PyTorch* MPS row-count bug regime. Our kernel
    must stay correct here (fp32 + kernel path)."""
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
    """Verify #4 MINOR: exercise the kernel's fp32 simd_sum reduction at D < 32 / not a multiple
    of 32 against an INDEPENDENT hand-computed fp32 oracle (no m._reference) so a subtle bug in the
    module's own reference could not mask a partial-simdgroup reduction error."""
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
    """Verify #3 MAJOR: when the F.rms_norm reroute bypasses the kernel on an outer guard (here a
    non-MPS device), it must set _last_backend = 'fallback' so a stale 'kernel' value from a previous
    successful call cannot create a false-positive kernel-path test. Uses a CPU tensor with a VALID
    normalized_shape so the bypass reaches the real stock rms_norm (which then succeeds)."""
    m.install_for_test()
    try:
        m._last_backend = "kernel"                        # simulate a stale spy from a prior kernel run
        x = torch.randn(4, 16)                            # CPU tensor -> device.type != "mps" outer bypass
        w = torch.randn(16)
        out = torch.nn.functional.rms_norm(x, (16,), w, 1e-6)
        assert m._last_backend == "fallback", "outer reroute bypass must reset the spy to 'fallback'"
        assert out.shape == x.shape
    finally:
        m.uninstall_for_test()
