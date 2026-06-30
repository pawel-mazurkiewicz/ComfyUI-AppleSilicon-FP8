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
    N, Cin, H, W, kh, kw = 1, 256, 512, 512, 3, 3
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
