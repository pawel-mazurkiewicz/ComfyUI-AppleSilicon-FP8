import pytest
import torch

from conftest import requires_mps  # repo-provided marker (tests/conftest.py)


@requires_mps
@pytest.mark.parametrize("H,W,Cin,kh,kw,s,p", [(16, 16, 8, 3, 3, 1, 1),
                                               (15, 17, 6, 3, 3, 2, 1),
                                               (8, 8, 4, 1, 1, 1, 0)])
def test_im2col_2d_matches_unfold(H, W, Cin, kh, kw, s, p):
    from _patches.conv_im2col_mps import _im2col_2d_full  # test helper: full (untiled) im2col
    torch.manual_seed(0)
    x = torch.randn(1, Cin, H, W, device="mps", dtype=torch.float16)
    # F.unfold gives [N, Cin*kh*kw, L] with the SAME (c,ki,kj) ordering as our K index.
    ref = torch.nn.functional.unfold(x.float(), (kh, kw), stride=s, padding=p)  # [1, K, P]
    ref = ref[0].t().contiguous()  # [P, K]
    A = _im2col_2d_full(x, kh, kw, s, p).float()  # [P, K]
    torch.mps.synchronize()
    assert A.shape == ref.shape, (A.shape, ref.shape)
    assert (A - ref).abs().max().item() < 1e-2


@requires_mps
@pytest.mark.parametrize("M,K,N,bias", [(130, 96, 72, False), (64, 256, 64, True),
                                        (300, 1152, 128, True)])
def test_gemm_nt_bias_matches_reference(M, K, N, bias, monkeypatch):
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import _gemm_nt_bias
    torch.manual_seed(0)
    # SPY: wrap the compiled lib so the test proves the real Metal entry point ran.
    # NOTE: torch.mps compiled-shader objects route attribute access to kernel-function
    # lookup (you cannot set/get arbitrary attrs on them), so we wrap the lib in a thin
    # delegating proxy that counts gemm_nt_bias invocations and forwards everything else.
    calls = {"gemm_nt_bias": 0}
    real_lib = cm._lib

    class _SpyLib:
        def __init__(self, inner):
            self._inner = inner

        def gemm_nt_bias(self, *a, **k):
            calls["gemm_nt_bias"] += 1
            return self._inner.gemm_nt_bias(*a, **k)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def spy_lib(dtype):
        return _SpyLib(real_lib(dtype))
    monkeypatch.setattr(cm, "_lib", spy_lib)

    A = torch.randn(M, K, device="mps", dtype=torch.float16)
    Bw = torch.randn(N, K, device="mps", dtype=torch.float16)
    b = torch.randn(N, device="mps", dtype=torch.float32) if bias else None
    out = torch.empty(M, N, device="mps", dtype=torch.float16)
    ref = A.float() @ Bw.float().t()
    if bias:
        ref = ref + b
    _gemm_nt_bias(A, Bw, b, out)          # writes directly into `out` (no per-call alloc)
    torch.mps.synchronize()
    out = out.float()
    assert calls["gemm_nt_bias"] == 1, "Metal gemm_nt_bias kernel did not run"
    assert out.shape == (M, N)
    # fp16 operands, fp32 accumulate: error is operand rounding only
    assert (out - ref).abs().max().item() < 2e-1


@requires_mps
@pytest.mark.parametrize("H,W,Cin,Cout,s,p,bias", [(64, 64, 32, 32, 1, 1, False),
                                                   (128, 128, 64, 128, 1, 1, True),
                                                   (65, 63, 16, 24, 2, 1, True)])
def test_conv2d_matches_reference(H, W, Cin, Cout, s, p, bias, monkeypatch):
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import conv_im2col
    torch.manual_seed(0)
    x = torch.randn(1, Cin, H, W, device="mps", dtype=torch.float16)
    w = torch.randn(Cout, Cin, 3, 3, device="mps", dtype=torch.float16)
    b = torch.randn(Cout, device="mps", dtype=torch.float16) if bias else None
    ref = torch.nn.functional.conv2d(x.float(), w.float(),
                                     b.float() if bias else None, stride=s, padding=p)
    stock = torch.nn.functional.conv2d(x, w, b, stride=s, padding=p).float()  # capture BEFORE spy
    # SPY: make the in-module stock fallback explode so a silent fallback fails the test.
    def boom(*a, **k):
        raise AssertionError("conv_im2col fell back to stock conv instead of running the kernel")
    monkeypatch.setattr(cm, "_fallback_conv", boom, raising=True)
    out = conv_im2col(x, w, b, stride=s, padding=p).float()
    torch.mps.synchronize()
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < 2e-1
    # must be no worse than stock fp16 conv
    assert (out - ref).abs().max().item() <= (stock - ref).abs().max().item() + 1e-2


@requires_mps
def test_tile_buffer_capped():
    """DETERMINISTIC OOM proof: the A_tile the driver allocates is provably <= _TILE_BYTES,
    and for a large conv it actually tiles (tile_p < P) rather than materializing full im2col."""
    import _patches.conv_im2col_mps as cm
    # 512x512, Cin=256, 3x3 -> full im2col is 1.21 GB, well above the 384 MB default cap.
    N, Cin, kh, kw = 1, 256, 3, 3  # H = W = 512
    Hout = Wout = 512  # pad=1, stride=1
    P, K = N * Hout * Wout, Cin * kh * kw
    elsize = 2  # fp16
    tile_p = max(1, min(P, cm._TILE_BYTES // (K * elsize)))
    a_tile_bytes = tile_p * K * elsize
    full_im2col_bytes = P * K * elsize          # = 262144*2304*2 = 1.21 GB
    assert a_tile_bytes <= cm._TILE_BYTES, (a_tile_bytes, cm._TILE_BYTES)
    assert tile_p < P, "must tile, not materialize full im2col"
    assert a_tile_bytes < full_im2col_bytes / 3   # tile is a small fraction of full im2col


@requires_mps
def test_conv_alloc_smoke_nonpeak():
    """NON-PEAK smoke: current_allocated delta with the output held live stays under an explicit
    budget = _TILE_BYTES + out_flat + contiguous-copy + weight + slack. (current_allocated is NOT
    a high-watermark; the deterministic guarantee is test_tile_buffer_capped above.)"""
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import conv_im2col
    torch.mps.empty_cache()
    x = torch.randn(1, 256, 512, 512, device="mps", dtype=torch.float16)
    w = torch.randn(256, 256, 3, 3, device="mps", dtype=torch.float16)
    base = torch.mps.current_allocated_memory()
    out = conv_im2col(x, w, None, 1, 1)       # keep `out` live so its bytes are counted
    torch.mps.synchronize()
    delta = torch.mps.current_allocated_memory() - base
    out_bytes = out.numel() * out.element_size()          # P*Cout*2
    weight_bytes = w.numel() * w.element_size()
    budget = cm._TILE_BYTES + 2 * out_bytes + weight_bytes + 64 * 1024 * 1024  # +64MB slack
    full_im2col_bytes = (512 * 512) * (256 * 9) * 2       # 1.21 GB
    assert delta < budget, (delta, budget)
    assert delta < full_im2col_bytes, (delta, full_im2col_bytes)


@requires_mps
@pytest.mark.parametrize("D,H,W,Cin,Cout,bias", [(5, 32, 32, 16, 16, False),
                                                 (4, 24, 24, 8, 12, True)])
def test_conv3d_matches_reference(D, H, W, Cin, Cout, bias, monkeypatch):
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import conv_im2col
    torch.manual_seed(0)
    x = torch.randn(1, Cin, D, H, W, device="mps", dtype=torch.float16)
    w = torch.randn(Cout, Cin, 3, 3, 3, device="mps", dtype=torch.float16)
    b = torch.randn(Cout, device="mps", dtype=torch.float16) if bias else None
    ref = torch.nn.functional.conv3d(x.float(), w.float(),
                                     b.float() if bias else None, padding=1)
    # SPY: stock fallback must not be how this test passes.
    def boom(*a, **k):
        raise AssertionError("conv3d fell back to stock conv instead of running the kernel")
    monkeypatch.setattr(cm, "_fallback_conv", boom, raising=True)
    out = conv_im2col(x, w, b, stride=1, padding=1).float()
    torch.mps.synchronize()
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < 2e-1


def test_install_noop_without_flag(monkeypatch):
    import _patches.conv_im2col_mps as cm
    monkeypatch.delenv("ASFP8_CONV_IM2COL", raising=False)
    monkeypatch.setattr(cm, "_installed_ranks", set(), raising=False)
    monkeypatch.setattr(cm, "_orig_conv2d", None, raising=False)
    cm.install()
    assert cm._installed_ranks == set()


def test_install_idempotent_per_mode(monkeypatch):
    """=2d then =3d in one process must install BOTH ranks."""
    import torch.nn.functional as F
    import _patches.conv_im2col_mps as cm
    monkeypatch.setattr(cm, "_installed_ranks", set(), raising=False)
    monkeypatch.setattr(cm, "_orig_conv2d", None, raising=False)
    monkeypatch.setattr(cm, "_orig_conv3d", None, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)
    monkeypatch.setenv("ASFP8_CONV_IM2COL", "2d")
    cm.install()
    assert cm._installed_ranks == {2}
    monkeypatch.setenv("ASFP8_CONV_IM2COL", "3d")
    cm.install()
    assert cm._installed_ranks == {2, 3}
    # restore F.conv2d/conv3d to the captured originals so we don't leak the wrap.
    F.conv2d, F.conv3d = cm._orig_conv2d, cm._orig_conv3d


@pytest.mark.parametrize("kwargs", [{"groups": 2}, {"dilation": 2}])
def test_wrapper_falls_back_unsupported(kwargs):
    """Grouped / dilated convs must delegate to the captured original, not the kernel."""
    import _patches.conv_im2col_mps as cm
    sentinel = object()
    called = {}

    def fake_orig(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        called["hit"] = True
        return sentinel
    conv2 = cm._make_wrap(fake_orig, 2)  # module-level wrapper factory (no MPS/install needed)
    x = torch.zeros(1, 4, 8, 8)          # CPU tensor, also unsupported device
    w = torch.zeros(8, 4, 3, 3)
    out = conv2(x, w, padding=1, **kwargs)
    assert out is sentinel and called.get("hit") is True


def test_wrapper_falls_back_on_kernel_exception(monkeypatch):
    """If the kernel raises, the wrapper must still return the original's result."""
    import _patches.conv_im2col_mps as cm
    sentinel = object()

    def fake_orig(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return sentinel

    def boom(*a, **k):
        raise RuntimeError("kernel blew up")
    monkeypatch.setattr(cm, "_supported", lambda *a, **k: True)
    monkeypatch.setattr(cm, "conv_im2col", boom)
    conv2 = cm._make_wrap(fake_orig, 2)
    out = conv2(torch.zeros(1, 4, 8, 8), torch.zeros(8, 4, 3, 3), padding=1)
    assert out is sentinel


def _boom_fallback(*a, **k):
    raise AssertionError("fell back to stock conv instead of running the Metal kernel")


@requires_mps
def test_fallback_after_install_no_recursion(monkeypatch):
    """[MAJOR #1] After install() swaps F.conv2d for the wrapper, a kernel failure inside
    conv_im2col must fall back to the captured ORIGINAL conv -- not re-enter the wrapper.
    Before the fix _fallback_conv called F.conv2d (= the wrapper) -> wrapper -> conv_im2col
    -> kernel raises -> _fallback_conv -> wrapper -> ... infinite recursion. We force the
    kernel to raise and assert exactly ONE kernel attempt (no re-entry) + a correct result."""
    import torch.nn.functional as F
    import _patches.conv_im2col_mps as cm
    saved2, saved3 = F.conv2d, F.conv3d
    try:
        # fresh install state, pretend MPS is installable, install conv2d wrapper
        monkeypatch.setattr(cm, "_installed_ranks", set(), raising=False)
        monkeypatch.setattr(cm, "_orig_conv2d", None, raising=False)
        monkeypatch.setattr(cm, "_orig_conv3d", None, raising=False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)
        monkeypatch.setenv("ASFP8_CONV_IM2COL", "2d")
        cm.install()
        assert cm._installed_ranks == {2}
        assert F.conv2d is not saved2, "install() did not replace F.conv2d with the wrapper"

        # force the kernel to raise so conv_im2col hits its INTERNAL fallback seam
        calls = {"checked": 0}

        def boom_checked(*a, **k):
            calls["checked"] += 1
            raise RuntimeError("forced kernel failure")
        monkeypatch.setattr(cm, "_conv2d_im2col_checked", boom_checked)

        x = torch.randn(1, 4, 8, 8, device="mps", dtype=torch.float16)
        w = torch.randn(8, 4, 3, 3, device="mps", dtype=torch.float16)
        ref = saved2(x, w, None, 1, 1).float()           # stock conv via TRUE original
        # call THROUGH the installed wrapper (what real callers hit after install)
        out = F.conv2d(x, w, None, 1, 1).float()
        torch.mps.synchronize()
        # with the recursion bug this is re-entered many times (or RecursionError); the
        # fix makes the fallback use the captured original -> exactly ONE kernel attempt.
        assert calls["checked"] == 1, calls
        assert out.shape == ref.shape
        assert (out - ref).abs().max().item() < 1e-2
    finally:
        F.conv2d, F.conv3d = saved2, saved3


@requires_mps
def test_conv3d_multitile_matches_reference(monkeypatch):
    """[MAJOR #2] Force tiny tiles so the conv3d tiling loop runs MANY tiles, and compare
    against the F.conv3d fp32 reference. All other correctness tests are single-tile
    (tile_p >= P); this is the only test that exercises the p0/rows tiling loop for real."""
    import math
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import conv_im2col
    torch.manual_seed(0)
    N, Cin, Cout, D, H, W = 1, 8, 12, 4, 12, 12
    x = torch.randn(N, Cin, D, H, W, device="mps", dtype=torch.float16)
    w = torch.randn(Cout, Cin, 3, 3, 3, device="mps", dtype=torch.float16)
    b = torch.randn(Cout, device="mps", dtype=torch.float16)
    P = N * D * H * W                       # 576 output pixels (pad=1, stride=1)
    K = Cin * 3 * 3 * 3                      # 216
    # cap A_tile to ~8 rows -> ~72 tiles, deep into the tiling loop
    tile_p_target = 8
    monkeypatch.setattr(cm, "_TILE_BYTES", tile_p_target * K * 2, raising=False)
    tile_p = max(1, min(P, cm._TILE_BYTES // (K * 2)))
    expected_tiles = math.ceil(P / tile_p)
    assert tile_p < P and expected_tiles > 1, (tile_p, P, expected_tiles)

    # count tiling-loop iterations to PROVE many tiles really ran
    n_tiles = {"n": 0}
    orig_tile = cm._im2col_3d_tile

    def spy_tile(*a, **k):
        n_tiles["n"] += 1
        return orig_tile(*a, **k)
    monkeypatch.setattr(cm, "_im2col_3d_tile", spy_tile)
    monkeypatch.setattr(cm, "_fallback_conv", _boom_fallback, raising=True)

    ref = torch.nn.functional.conv3d(x.float(), w.float(), b.float(), padding=1)
    out = conv_im2col(x, w, b, stride=1, padding=1).float()
    torch.mps.synchronize()
    assert n_tiles["n"] == expected_tiles, (n_tiles, expected_tiles)
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < 2e-1


@requires_mps
@pytest.mark.parametrize("rank", [2, 3])
def test_conv_batch_n_gt_1(rank, monkeypatch):
    """[MINOR #5] N>1 batch: the pixel decode n = pix/(...) must place batches correctly."""
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import conv_im2col
    torch.manual_seed(0)
    monkeypatch.setattr(cm, "_fallback_conv", _boom_fallback, raising=True)
    if rank == 2:
        x = torch.randn(3, 8, 16, 16, device="mps", dtype=torch.float16)
        w = torch.randn(12, 8, 3, 3, device="mps", dtype=torch.float16)
        ref = torch.nn.functional.conv2d(x.float(), w.float(), padding=1)
    else:
        x = torch.randn(2, 6, 4, 12, 12, device="mps", dtype=torch.float16)
        w = torch.randn(10, 6, 3, 3, 3, device="mps", dtype=torch.float16)
        ref = torch.nn.functional.conv3d(x.float(), w.float(), padding=1)
    out = conv_im2col(x, w, None, stride=1, padding=1).float()
    torch.mps.synchronize()
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < 2e-1


@requires_mps
def test_conv3d_stride2_matches_reference(monkeypatch):
    """[MINOR #6] conv3d with stride>1 (sD/sH/sW used independently in the gather)."""
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import conv_im2col
    torch.manual_seed(0)
    monkeypatch.setattr(cm, "_fallback_conv", _boom_fallback, raising=True)
    x = torch.randn(1, 8, 7, 17, 15, device="mps", dtype=torch.float16)
    w = torch.randn(10, 8, 3, 3, 3, device="mps", dtype=torch.float16)
    b = torch.randn(10, device="mps", dtype=torch.float16)
    ref = torch.nn.functional.conv3d(x.float(), w.float(), b.float(), stride=2, padding=1)
    out = conv_im2col(x, w, b, stride=2, padding=1).float()
    torch.mps.synchronize()
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < 2e-1


@requires_mps
def test_conv3d_bf16_end_to_end(monkeypatch):
    """[MINOR #4] bf16 end-to-end conv (all other conv tests are fp16). Result must be no
    worse than the stock bf16 conv against the fp32 reference."""
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import conv_im2col
    torch.manual_seed(0)
    x = torch.randn(1, 6, 4, 12, 12, device="mps", dtype=torch.bfloat16)
    w = torch.randn(8, 6, 3, 3, 3, device="mps", dtype=torch.bfloat16)
    b = torch.randn(8, device="mps", dtype=torch.bfloat16)
    ref = torch.nn.functional.conv3d(x.float(), w.float(), b.float(), padding=1)
    stock = torch.nn.functional.conv3d(x, w, b, padding=1).float()  # capture BEFORE spy
    monkeypatch.setattr(cm, "_fallback_conv", _boom_fallback, raising=True)
    out = conv_im2col(x, w, b, stride=1, padding=1).float()
    torch.mps.synchronize()
    assert out.dtype == torch.float32 and ref.shape == out.shape
    # no worse than stock bf16 conv against the fp32 reference (+ small slack)
    assert (out - ref).abs().max().item() <= (stock - ref).abs().max().item() + 5e-2


def test_supported_rejects_dtype_mismatch():
    """[MINOR #7] _supported must reject weight.dtype != x.dtype (kernel is compiled for the
    input dtype and would read the weight buffer's bytes as the wrong type -> garbage)."""
    import _patches.conv_im2col_mps as cm

    class _FakeT:
        def __init__(self, dt):
            self.dtype = dt
            self.device = type("D", (), {"type": "mps"})()
    x = _FakeT(torch.float16)
    w_ok = _FakeT(torch.float16)
    w_bad = _FakeT(torch.float32)
    assert cm._supported(x, w_ok, 1, 1, x.dtype) is True
    assert cm._supported(x, w_bad, 1, 1, x.dtype) is False


@requires_mps
def test_scatter_matches_nonscatter(monkeypatch):
    """The fused channel-major scatter epilogue must produce byte-identical output to the
    out_flat+permute path (same kernel math, different store)."""
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import conv_im2col
    torch.manual_seed(0)
    x = torch.randn(1, 16, 5, 24, 24, device="mps", dtype=torch.float16)
    w = torch.randn(12, 16, 3, 3, 3, device="mps", dtype=torch.float16)
    b = torch.randn(12, device="mps", dtype=torch.float16)
    # [MINOR #3] defense-in-depth: a silent fallback would make scatter==nonscatter trivially
    monkeypatch.setattr(cm, "_fallback_conv", _boom_fallback, raising=True)
    monkeypatch.setenv("ASFP8_CONV_SCATTER", "0")
    ref = conv_im2col(x, w, b, stride=1, padding=1)
    monkeypatch.setenv("ASFP8_CONV_SCATTER", "1")
    got = conv_im2col(x, w, b, stride=1, padding=1)
    torch.mps.synchronize()
    assert got.shape == ref.shape
    assert torch.equal(got, ref), (got - ref).abs().max().item()


@requires_mps
def test_scatter_drops_extra_buffer(monkeypatch):
    """DETERMINISTIC proof (current_allocated is NOT a high-watermark, so it cannot observe
    the transient out_flat/copy that scatter removes). We spy torch.empty: the scatter path
    must allocate ONLY the channel-major final output [N,Cout,H,W] + the [tile_p,K] A_tile,
    and NEVER the [P,Cout] out_flat staging buffer the non-scatter path allocates."""
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import conv_im2col
    # [MINOR #3] defense-in-depth: stock fallback doesn't allocate [P,Cout] either, so a
    # silent fallback could mask a broken scatter path -- make it explode instead.
    monkeypatch.setattr(cm, "_fallback_conv", _boom_fallback, raising=True)
    N, Cin, H, W, Cout = 1, 64, 256, 256, 128
    P = N * H * W  # pad=1, stride=1 -> Hout=H, Wout=W
    x = torch.randn(N, Cin, H, W, device="mps", dtype=torch.float16)
    w = torch.randn(Cout, Cin, 3, 3, device="mps", dtype=torch.float16)

    real_empty = torch.empty

    def run_and_capture(scatter):
        shapes = []

        def spy_empty(*size, **kw):
            # torch.empty(*sizes) or torch.empty((sizes,))
            s = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
            shapes.append(tuple(int(d) for d in s))
            return real_empty(*size, **kw)
        monkeypatch.setenv("ASFP8_CONV_SCATTER", "1" if scatter else "0")
        monkeypatch.setattr(torch, "empty", spy_empty)
        out = conv_im2col(x, w, None, 1, 1)
        torch.mps.synchronize()
        monkeypatch.setattr(torch, "empty", real_empty)
        return shapes, out

    s_scatter, out = run_and_capture(True)
    s_nonscatter, _ = run_and_capture(False)

    out_flat_shape = (P, Cout)
    final_shape = (N, Cout, H, W)
    # non-scatter allocates the [P,Cout] out_flat staging buffer ...
    assert out_flat_shape in s_nonscatter, s_nonscatter
    # ... scatter NEVER does; it allocates the channel-major output directly.
    assert out_flat_shape not in s_scatter, s_scatter
    assert final_shape in s_scatter, s_scatter
    assert out.shape == final_shape
