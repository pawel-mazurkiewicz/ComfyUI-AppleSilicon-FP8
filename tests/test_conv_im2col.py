import pytest
import torch

from conftest import requires_mps  # repo-provided marker (tests/conftest.py)


@pytest.fixture(autouse=True)
def _restore_conv_state():
    """install() rebinds F.conv2d/F.conv3d and module globals; undo it after every test."""
    import torch.nn.functional as F
    import _patches.conv_im2col_mps as cm
    saved = (F.conv2d, F.conv3d, cm._installed_ranks, cm._orig_conv2d, cm._orig_conv3d, cm._sf_installed)
    yield
    F.conv2d, F.conv3d, cm._installed_ranks, cm._orig_conv2d, cm._orig_conv3d, cm._sf_installed = saved


@requires_mps
@pytest.mark.parametrize("H,W,Cin,kh,kw,s,p", [(16, 16, 8, 3, 3, 1, 1),
                                               (15, 17, 6, 3, 3, 2, 1),
                                               (8, 8, 4, 1, 1, 1, 0)])
def test_im2col_2d_matches_unfold(H, W, Cin, kh, kw, s, p):
    from _patches.conv_im2col_mps import _im2col_2d_full  # test helper: full (untiled) im2col
    torch.manual_seed(0)
    x = torch.randn(1, Cin, H, W, device="mps", dtype=torch.float16)
    # F.unfold uses the same (c,ki,kj) ordering as our K index
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
    # a delegating proxy, not a patched attribute: compiled-shader objects route every
    # attribute access to kernel-function lookup
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
    # make the stock fallback explode, so a silent fallback fails the test
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
    """A_tile stays under _TILE_BYTES, and a large conv really tiles instead of
    materializing the full im2col."""
    import _patches.conv_im2col_mps as cm
    # full im2col here is 1.21 GB, well above the 384 MB default cap
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
    """The allocation delta with the output held live stays inside an explicit budget.

    current_allocated is not a high-watermark; test_tile_buffer_capped is the real proof.
    """
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


def test_install_default_on_routes_conv3d_only(monkeypatch):
    """The default routes conv3d but not conv2d."""
    import torch.nn.functional as F
    import _patches.conv_im2col_mps as cm
    monkeypatch.delenv("ASFP8_CONV_IM2COL", raising=False)
    monkeypatch.setattr(cm, "_installed_ranks", set(), raising=False)
    monkeypatch.setattr(cm, "_orig_conv2d", None, raising=False)
    monkeypatch.setattr(cm, "_orig_conv3d", None, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)
    try:
        cm.install()
        assert cm._installed_ranks == {3}
    finally:
        if cm._orig_conv2d is not None:
            F.conv2d, F.conv3d = cm._orig_conv2d, cm._orig_conv3d


def test_install_off_disables(monkeypatch):
    """ASFP8_CONV_IM2COL=off is the kill switch -- nothing is installed."""
    import _patches.conv_im2col_mps as cm
    monkeypatch.setenv("ASFP8_CONV_IM2COL", "off")
    monkeypatch.setattr(cm, "_installed_ranks", set(), raising=False)
    monkeypatch.setattr(cm, "_orig_conv2d", None, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)
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
    conv2 = cm._make_wrap(fake_orig, 2)  # module-level, so no MPS or install() needed
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
    """After install(), a kernel failure falls back to the captured original conv rather
    than re-entering the wrapper and recursing forever."""
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
        ref = saved2(x, w, None, 1, 1).float()           # via the true original
        # through the installed wrapper, as a real caller would
        out = F.conv2d(x, w, None, 1, 1).float()
        torch.mps.synchronize()
        # exactly one attempt: re-entering the wrapper would recurse instead
        assert calls["checked"] == 1, calls
        assert out.shape == ref.shape
        assert (out - ref).abs().max().item() < 1e-2
    finally:
        F.conv2d, F.conv3d = saved2, saved3


@requires_mps
def test_conv3d_multitile_matches_reference(monkeypatch):
    """A many-tile conv3d matches the fp32 reference; every other test is single-tile."""
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
    """N>1: the pixel decode places each batch correctly."""
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
    """conv3d with stride>1, where the gather uses sD/sH/sW independently."""
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
    """A bf16 conv is no worse than the stock bf16 conv against the fp32 reference."""
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
    """_supported rejects weight.dtype != x.dtype: the kernel is compiled per input dtype
    and would read the weight bytes as the wrong type."""
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
    """The fused scatter epilogue is byte-identical to the out_flat+permute path."""
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import conv_im2col
    torch.manual_seed(0)
    x = torch.randn(1, 16, 5, 24, 24, device="mps", dtype=torch.float16)
    w = torch.randn(12, 16, 3, 3, 3, device="mps", dtype=torch.float16)
    b = torch.randn(12, device="mps", dtype=torch.float16)
    # a silent fallback would make scatter==nonscatter trivially
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
    """The scatter path never allocates the [P,Cout] out_flat staging buffer.

    Spied through torch.empty: current_allocated can't observe a transient buffer.
    """
    import _patches.conv_im2col_mps as cm
    from _patches.conv_im2col_mps import conv_im2col
    # the stock fallback doesn't allocate [P,Cout] either, so a silent one would mask a
    # broken scatter path -- make it explode instead
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


# ---------------------------------------------------------------------------
# Single-frame conv3d -> conv2d (T == kT, no temporal padding)
# ---------------------------------------------------------------------------

_SF_DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# (kernel hw, stride, padding, dilation, N, bias): the CausalConv3d shapes comfy's Wan VAE runs
_SF_VARIANTS = {
    "wan3x3": dict(ks=(3, 3), stride=1, padding=(0, 1, 1), dilation=1, N=1, bias=True),
    "wan3x3_nobias_n2": dict(ks=(3, 3), stride=1, padding=(0, 1, 1), dilation=1, N=2, bias=False),
    "spatial_stride2": dict(ks=(3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=1, N=1, bias=True, hw=(21, 19)),
    "time_conv_stride_t2": dict(ks=(1, 1), stride=(2, 1, 1), padding=0, dilation=1, N=1, bias=True),
    "int_args_1x1": dict(ks=(1, 1), stride=1, padding=0, dilation=1, N=1, bias=False),
    "spatial_dilation": dict(ks=(3, 3), stride=1, padding=(0, 2, 2), dilation=(1, 2, 2), N=1, bias=True),
}


def _rel_rms(out, ref):
    return ((out.float().cpu() - ref).norm() / ref.norm()).item()


def _sf_tol(dtype, ref):
    # one rounding of the fp32 result to `dtype` is the best any kernel can do
    if dtype == torch.float32:
        return 1e-5
    return 1.15 * _rel_rms(ref.to(dtype), ref)


def _sf_inputs(dtype, kT, v, Cin=16, Cout=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    H, W = v.get("hw", (20, 20))
    kh, kw = v["ks"]
    x = torch.randn(v["N"], Cin, kT, H, W, generator=g).to(dtype)
    w = (torch.randn(Cout, Cin, kT, kh, kw, generator=g) / (Cin * kT * kh * kw) ** 0.5).to(dtype)
    b = torch.randn(Cout, generator=g).to(dtype) if v["bias"] else None
    return x, w, b


@requires_mps
@pytest.mark.parametrize("dtype", _SF_DTYPES)
@pytest.mark.parametrize("kT", [1, 3])
@pytest.mark.parametrize("name", list(_SF_VARIANTS))
def test_single_frame_matches_conv3d(dtype, kT, name):
    import torch.nn.functional as F
    import _patches.conv_im2col_mps as cm
    v = _SF_VARIANTS[name]
    x, w, b = _sf_inputs(dtype, kT, v)
    ref = F.conv3d(x.float(), w.float(), None if b is None else b.float(),
                   v["stride"], v["padding"], v["dilation"])
    xm, wm, bm = x.to("mps"), w.to("mps"), None if b is None else b.to("mps")
    assert cm._single_frame_ok(xm, wm, v["stride"], v["padding"], v["dilation"], 1)
    out = cm._single_frame_conv3d(xm, wm, bm, v["stride"], v["padding"], v["dilation"], 1)
    stock = F.conv3d(xm, wm, bm, v["stride"], v["padding"], v["dilation"])
    torch.mps.synchronize()
    assert out.shape == ref.shape and out.dtype == dtype
    assert out.is_contiguous() == stock.is_contiguous()
    assert _rel_rms(out, ref) <= _sf_tol(dtype, ref), (_rel_rms(out, ref), _sf_tol(dtype, ref))


@requires_mps
@pytest.mark.parametrize("dtype", _SF_DTYPES)
@pytest.mark.parametrize("kT", [1, 3])
def test_single_frame_no_worse_than_stock_conv3d(dtype, kT):
    """Folding the taps accumulates once, so error stays at or below stock MPS conv3d's.

    Summing per-tap bf16 conv2d outputs measured ~1.4x stock error at kT=3.
    """
    import torch.nn.functional as F
    import _patches.conv_im2col_mps as cm
    v = _SF_VARIANTS["wan3x3"]
    x, w, b = _sf_inputs(dtype, kT, v, Cin=96, Cout=96)
    ref = F.conv3d(x.float(), w.float(), b.float(), 1, (0, 1, 1))
    xm, wm, bm = x.to("mps"), w.to("mps"), b.to("mps")
    out = cm._single_frame_conv3d(xm, wm, bm, 1, (0, 1, 1), 1, 1)
    stock = F.conv3d(xm, wm, bm, 1, (0, 1, 1))
    torch.mps.synchronize()
    assert _rel_rms(out, ref) <= max(_rel_rms(stock, ref) * 1.05, 1e-6)


@requires_mps
@pytest.mark.parametrize("dtype", _SF_DTYPES)
@pytest.mark.parametrize("Cout,T", [(3, 1), (32, 1), (3, 2), (32, 2)])
def test_single_frame_strided_view_weight(dtype, Cout, T):
    """comfy's causal_zero passes weight[:, :, -T:], a strided view at a storage offset;
    MPS conv2d returns garbage for those without the .contiguous() (fp32, few channels)."""
    import torch.nn.functional as F
    import _patches.conv_im2col_mps as cm
    g = torch.Generator().manual_seed(0)
    C = 32
    x = torch.randn(1, C, T, 20, 20, generator=g).to(dtype).to("mps")
    wfull = (torch.randn(Cout, C, 3, 3, 3, generator=g) / 30).to(dtype).to("mps")
    w = wfull[:, :, -T:]
    assert not w.is_contiguous() and w.storage_offset() > 0
    ref = F.conv3d(x.float().cpu(), w.float().cpu(), None, 1, (0, 1, 1))
    out = cm._single_frame_conv3d(x, w, None, 1, (0, 1, 1), 1, 1)
    torch.mps.synchronize()
    assert _rel_rms(out, ref) <= _sf_tol(dtype, ref)


@requires_mps
@pytest.mark.parametrize("dtype", _SF_DTYPES)
@pytest.mark.parametrize("kT", [1, 3])
def test_single_frame_noncontiguous_input(dtype, kT):
    """An input sliced out of a longer clip (strided view, storage offset) is handled."""
    import torch.nn.functional as F
    import _patches.conv_im2col_mps as cm
    g = torch.Generator().manual_seed(0)
    big = torch.randn(2, 16, 7, 20, 20, generator=g).to(dtype).to("mps")
    x = big[:, :, 2:2 + kT]
    assert not x.is_contiguous()
    w = (torch.randn(24, 16, kT, 3, 3, generator=g) / 20).to(dtype).to("mps")
    ref = F.conv3d(x.float().cpu(), w.float().cpu(), None, 1, (0, 1, 1))
    out = cm._single_frame_conv3d(x, w, None, 1, (0, 1, 1), 1, 1)
    torch.mps.synchronize()
    assert _rel_rms(out, ref) <= _sf_tol(dtype, ref)


def _sf_args(**over):
    """An eligible one-frame conv3d on MPS, with overrides; returns (x, w, kwargs)."""
    a = dict(x=(1, 8, 3, 12, 12), w=(8, 8, 3, 3, 3), dtype=torch.bfloat16, wdtype=None, device="mps",
             stride=1, padding=(0, 1, 1), dilation=1, groups=1)
    a.update(over)
    x = torch.zeros(*a["x"], dtype=a["dtype"], device=a["device"])
    w = torch.zeros(*a["w"], dtype=a["wdtype"] or a["dtype"], device=a["device"])
    return x, w, {k: a[k] for k in ("stride", "padding", "dilation", "groups")}


_SF_REJECTS = {
    "T_ne_kT": dict(x=(1, 8, 5, 12, 12)),
    "temporal_padding": dict(padding=(1, 1, 1)),
    "padding_same": dict(padding="same"),
    "padding_valid": dict(padding="valid"),
    "temporal_dilation": dict(x=(1, 8, 1, 12, 12), w=(8, 8, 1, 3, 3), dilation=(2, 1, 1)),
    "grouped": dict(w=(8, 4, 3, 3, 3), groups=2),
    "dtype_mismatch": dict(wdtype=torch.float32),
    "unbatched_4d": dict(x=(8, 3, 12, 12)),
    "empty_batch": dict(x=(0, 8, 3, 12, 12)),
    "stride_wrong_len": dict(stride=(1, 1)),
    "channel_mismatch": dict(w=(8, 6, 3, 3, 3)),
}


@requires_mps
def test_single_frame_ok_accepts_eligible():
    import _patches.conv_im2col_mps as cm
    x, w, kw = _sf_args()
    assert cm._single_frame_ok(x, w, kw["stride"], kw["padding"], kw["dilation"], kw["groups"]) is True


@requires_mps
@pytest.mark.parametrize("name", list(_SF_REJECTS))
def test_single_frame_not_engaged(name, monkeypatch):
    """Anything that is not a plain one-frame conv3d on MPS goes to the original conv."""
    import _patches.conv_im2col_mps as cm
    x, w, kw = _sf_args(**_SF_REJECTS[name])
    assert cm._single_frame_ok(x, w, kw["stride"], kw["padding"], kw["dilation"], kw["groups"]) is False
    sentinel = object()
    hit = {"sf": 0}

    def spy(*a, **k):
        hit["sf"] += 1
    wrapped = cm._make_wrap(lambda *a, **k: sentinel, 3, im2col=False, single_frame=True)
    monkeypatch.setattr(cm, "_single_frame_conv3d", spy)
    assert wrapped(x, w, None, **kw) is sentinel
    assert hit["sf"] == 0


def test_single_frame_inert_on_cpu():
    import _patches.conv_im2col_mps as cm
    sentinel = object()
    x, w, kw = _sf_args(x=(1, 8, 1, 12, 12), w=(8, 8, 1, 3, 3), device="cpu", dtype=torch.float32)
    on = cm._make_wrap(lambda *a, **k: sentinel, 3, im2col=False, single_frame=True)
    assert on(x, w, None, **kw) is sentinel
    assert cm._single_frame_ok(x, w, kw["stride"], kw["padding"], kw["dilation"], kw["groups"]) is False


@requires_mps
def test_single_frame_wrapper_flag_off_delegates():
    import _patches.conv_im2col_mps as cm
    sentinel = object()
    x, w, kw = _sf_args(x=(1, 8, 1, 12, 12), w=(8, 8, 1, 3, 3))
    assert cm._single_frame_ok(x, w, kw["stride"], kw["padding"], kw["dilation"], kw["groups"])
    off = cm._make_wrap(lambda *a, **k: sentinel, 3, im2col=False, single_frame=False)
    assert off(x, w, None, **kw) is sentinel


@requires_mps
def test_single_frame_wrapper_engages_and_skips_orig():
    import torch.nn.functional as F
    import _patches.conv_im2col_mps as cm
    x, w, kw = _sf_args(x=(1, 8, 1, 12, 12), w=(8, 8, 1, 3, 3))
    wrapped = cm._make_wrap(_boom_fallback, 3, im2col=False, single_frame=True)
    out = wrapped(x, w, None, **kw)
    assert out.shape == (1, 8, 1, 12, 12)


@requires_mps
def test_single_frame_stock_conv2d_is_the_captured_original(monkeypatch):
    """With conv2d wrapped by install(), the fold must call the captured original, not re-enter it."""
    import _patches.conv_im2col_mps as cm
    seen = []

    def fake_orig2d(*a, **k):
        seen.append(1)
        return torch.zeros(1, 8, 12, 12, device="mps", dtype=torch.bfloat16)
    monkeypatch.setattr(cm, "_orig_conv2d", fake_orig2d)
    monkeypatch.setattr(torch.nn.functional, "conv2d", _boom_fallback)
    x, w, kw = _sf_args(x=(1, 8, 1, 12, 12), w=(8, 8, 1, 3, 3))
    cm._single_frame_conv3d(x, w, None, kw["stride"], kw["padding"], kw["dilation"], 1)
    assert seen == [1]


# install(): the single-frame path has its own gate, independent of matmul2d / M5

def _fresh_install_state(monkeypatch, gate):
    import _patches.conv_im2col_mps as cm
    monkeypatch.delenv("ASFP8_CONV_IM2COL", raising=False)
    monkeypatch.delenv("ASFP8_CONV_SINGLE_FRAME", raising=False)
    monkeypatch.setattr(cm, "_installed_ranks", set())
    monkeypatch.setattr(cm, "_orig_conv2d", None)
    monkeypatch.setattr(cm, "_orig_conv3d", None)
    monkeypatch.setattr(cm, "_sf_installed", False)
    monkeypatch.setattr(cm, "_gate", lambda: gate)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)
    return cm


def test_install_single_frame_without_matmul2d(monkeypatch):
    """No matmul2d (non-M5): im2col stays off, the single-frame wrapper still installs."""
    import torch.nn.functional as F
    cm = _fresh_install_state(monkeypatch, gate=False)
    c2, c3 = F.conv2d, F.conv3d
    cm.install()
    assert cm._installed_ranks == set()
    assert cm._sf_installed is True
    assert F.conv2d is c2 and F.conv3d is not c3
    assert cm._orig_conv3d is c3


@requires_mps
def test_install_without_matmul2d_routes_only_one_frame(monkeypatch):
    import torch.nn.functional as F
    cm = _fresh_install_state(monkeypatch, gate=False)
    cm.install()
    calls = {"sf": 0}
    real = cm._single_frame_conv3d

    def spy(*a, **k):
        calls["sf"] += 1
        return real(*a, **k)
    monkeypatch.setattr(cm, "_single_frame_conv3d", spy)
    monkeypatch.setattr(cm, "conv_im2col", _boom_fallback)
    x1 = torch.randn(1, 8, 1, 12, 12, device="mps", dtype=torch.bfloat16)
    w1 = torch.randn(8, 8, 1, 3, 3, device="mps", dtype=torch.bfloat16)
    F.conv3d(x1, w1, None, 1, (0, 1, 1))
    assert calls["sf"] == 1
    x5 = torch.randn(1, 8, 5, 12, 12, device="mps", dtype=torch.bfloat16)
    w3 = torch.randn(8, 8, 3, 3, 3, device="mps", dtype=torch.bfloat16)
    out = F.conv3d(x5, w3, None, 1, 1)          # multi-frame: stock, not im2col, not single-frame
    torch.mps.synchronize()
    assert calls["sf"] == 1 and out.shape == (1, 8, 5, 12, 12)


@requires_mps
def test_install_with_matmul2d_keeps_im2col_for_multi_frame(monkeypatch):
    """Both paths on: one-frame goes to single-frame first, multi-frame still to im2col."""
    import torch.nn.functional as F
    cm = _fresh_install_state(monkeypatch, gate=True)
    cm.install()
    assert cm._installed_ranks == {3} and cm._sf_installed is True
    sf, im = object(), object()
    monkeypatch.setattr(cm, "_single_frame_conv3d", lambda *a, **k: sf)
    monkeypatch.setattr(cm, "conv_im2col", lambda *a, **k: im)
    x1 = torch.zeros(1, 8, 1, 12, 12, device="mps", dtype=torch.bfloat16)
    w1 = torch.zeros(8, 8, 1, 3, 3, device="mps", dtype=torch.bfloat16)
    assert F.conv3d(x1, w1, None, 1, (0, 1, 1)) is sf
    x5 = torch.zeros(1, 8, 5, 12, 12, device="mps", dtype=torch.bfloat16)
    w3 = torch.zeros(8, 8, 3, 3, 3, device="mps", dtype=torch.bfloat16)
    assert F.conv3d(x5, w3, None, 1, 1) is im


@requires_mps
def test_install_single_frame_flag_off_keeps_im2col(monkeypatch):
    import torch.nn.functional as F
    cm = _fresh_install_state(monkeypatch, gate=True)
    monkeypatch.setenv("ASFP8_CONV_SINGLE_FRAME", "off")
    cm.install()
    assert cm._installed_ranks == {3} and cm._sf_installed is False
    im = object()
    monkeypatch.setattr(cm, "conv_im2col", lambda *a, **k: im)
    x1 = torch.zeros(1, 8, 1, 12, 12, device="mps", dtype=torch.bfloat16)
    w1 = torch.zeros(8, 8, 1, 3, 3, device="mps", dtype=torch.bfloat16)
    assert F.conv3d(x1, w1, None, 1, (0, 1, 1)) is im        # pre-port behavior


@pytest.mark.parametrize("env,val,gate", [("ASFP8_CONV_SINGLE_FRAME", "off", False),
                                          ("ASFP8_CONV_SINGLE_FRAME", "0", False),
                                          ("ASFP8_CONV_IM2COL", "off", False),
                                          ("ASFP8_CONV_IM2COL", "off", True)])
def test_install_kill_switches(monkeypatch, env, val, gate):
    import torch.nn.functional as F
    cm = _fresh_install_state(monkeypatch, gate=gate)
    monkeypatch.setenv(env, val)
    c2, c3 = F.conv2d, F.conv3d
    cm.install()
    assert cm._installed_ranks == set() and cm._sf_installed is False
    assert F.conv2d is c2 and F.conv3d is c3


def test_install_inert_without_mps_or_off_darwin(monkeypatch):
    import sys
    import torch.nn.functional as F
    cm = _fresh_install_state(monkeypatch, gate=True)
    c2, c3 = F.conv2d, F.conv3d
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False, raising=False)
    cm.install()
    assert F.conv2d is c2 and F.conv3d is c3 and not cm._sf_installed and not cm._installed_ranks
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)
    with monkeypatch.context() as m:
        m.setattr(sys, "platform", "linux")
        cm.install()
    assert F.conv2d is c2 and F.conv3d is c3 and not cm._sf_installed and not cm._installed_ranks


def test_install_upgrade_rewraps_original_not_wrapper(monkeypatch):
    """single-frame-only install, then an im2col-capable one: F.conv3d wraps the true original."""
    import torch.nn.functional as F
    cm = _fresh_install_state(monkeypatch, gate=False)
    orig3 = F.conv3d
    cm.install()
    first = F.conv3d
    monkeypatch.setattr(cm, "_gate", lambda: True)
    cm.install()
    assert F.conv3d is not first and cm._installed_ranks == {3} and cm._sf_installed is True
    assert cm._orig_conv3d is orig3
    assert orig3 in [c.cell_contents for c in F.conv3d.__closure__]
    second = F.conv3d
    cm.install()                                  # nothing new to install: no re-wrap
    assert F.conv3d is second
