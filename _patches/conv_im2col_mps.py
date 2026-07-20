"""Patch #18: VAE conv on MPS via tiled im2col + matmul2d tensor-op GEMM (conv3d default-ON).

Routes conv2d/conv3d through `im2col_2d/3d` gather -> NT `matmul2d` GEMM (half/bf16/fp32
operands, fp32 cooperative-tensor accumulate) + fused bias, looping over output-pixel
tiles so the lowered patch buffer (`A_tile`) is capped (default 384 MB). Targets the
Session-15 SeedVR2 non-tiled conv3d decode OOM. Authored with torch.mps.compile_shader
(pure-Python authoring path, no .mm/xcrun/ninja build step).

Gating: conv3d is ON by default (im2col is ~2.7x faster than stock MPS conv3d and ~31%
faster end-to-end on SeedVR2 — measured on M5/Metal 4.1). conv2d stays OFF by default
(stock conv2d is already at-roofline; im2col loses there). Kill switch: ASFP8_CONV_IM2COL=off.
Opt conv2d in with =2d or =2d,3d (=1|on|true => both ranks). Build is M5/Metal-4.1 gated and
never fatal: any compile/kernel/shape failure falls back to stock F.conv2d/conv3d.

B.0 probe (M5 Max / macOS 27 / PyTorch 2.11 / Metal 4.1) confirmed the <T,T,float>
cooperative-accumulate matmul2d compiles+runs+CORRECT for half, bfloat AND float, so all
three are exposed below.
"""
import os

import torch
import torch.nn.functional as F

TAG = "[AppleSilicon-FP8/conv]"

def _tile_mb():
    # Parsed at import time; a malformed value must NOT crash the whole node before the
    # per-patch install guards run — fall back to the 384 MB default.
    try:
        return max(1, int(os.environ.get("ASFP8_CONV_TILE_MB", "384")))
    except (TypeError, ValueError):
        return 384


_TILE_BYTES = _tile_mb() * 1024 * 1024

# B.0 (dev/probe_matmul2d_dtype_scatter.py) recorded PASS for all three operand dtypes
# with an fp32 cooperative-tensor accumulator (matching the handed-down G2 result).
_DT = {
    torch.float16: "half",
    torch.bfloat16: "bfloat",
    torch.float32: "float",
}

# ---------------------------------------------------------------------------
# Metal sources
# ---------------------------------------------------------------------------

_IM2COL_2D_SRC = r"""
#include <metal_stdlib>
using namespace metal;
kernel void im2col_2d(
    device const @T@* X     [[buffer(0)]],
    device @T@*       Atile [[buffer(1)]],
    device const int* PRM   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const int Cin=PRM[1], H=PRM[2], W=PRM[3], kh=PRM[4], kw=PRM[5];
    const int sH=PRM[6], sW=PRM[7], pH=PRM[8], pW=PRM[9];
    const int Hout=PRM[10], Wout=PRM[11], K=PRM[12], p0=PRM[13], rows=PRM[14];
    const uint total = uint(rows) * uint(K);
    if (gid >= total) return;
    const int r=int(gid)/K, kk=int(gid)%K, pix=p0+r;
    const int ow=pix%Wout, oh=(pix/Wout)%Hout, n=pix/(Wout*Hout);
    const int kj=kk%kw, ki=(kk/kw)%kh, c=kk/(kh*kw);
    const int ih=oh*sH+ki-pH, iw=ow*sW+kj-pW;
    @T@ v=@T@(0);
    if (ih>=0 && ih<H && iw>=0 && iw<W)
        v = X[(((ulong(n)*Cin + c)*H + ih)*W + iw)];
    Atile[gid]=v;
}
"""

_GEMM_SRC = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant constexpr int BM = 64, BN = 64, NSG = 4;
kernel void gemm_nt_bias(
    device @T@*       A    [[buffer(0)]],
    device @T@*       Bw   [[buffer(1)]],
    device float*     BIAS [[buffer(2)]],
    device @T@*       OUT  [[buffer(3)]],
    device const int* PRM  [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    const int M=PRM[0], N=PRM[1], K=PRM[2], has_bias=PRM[3];
    const int m0=int(tgid.x)*BM, n0=int(tgid.y)*BN;
    if (m0>=M || n0>=N) return;
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto mA = tensor<device @T@, dextents<int,2>, tensor_inline>(
                  A  + ulong(m0)*K, dextents<int,2>{K, min(BM, M-m0)}, array<int,2>{1, K});
    auto mB = tensor<device @T@, dextents<int,2>, tensor_inline>(
                  Bw + ulong(n0)*K, dextents<int,2>{K, min(BN, N-n0)}, array<int,2>{1, K});
    using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(mA)>;
    using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(mB)>;
    auto cC = op.get_destination_cooperative_tensor<AT, BT, float>();
    for (uint16_t i=0;i<cC.get_capacity();++i) if (cC.is_valid_element(i)) cC[i]=0.0f;
    op.run(mA, mB, cC);
    device @T@* Cb = OUT + ulong(m0)*N + n0;
    for (uint16_t i=0;i<cC.get_capacity();++i){
        if(!cC.is_valid_element(i)) continue;
        auto idx=cC.get_multidimensional_index(i);
        const int r=int(idx[1]), c=int(idx[0]);
        if(m0+r>=M || n0+c>=N) continue;
        float y=cC[i];
        if(has_bias) y += BIAS[n0+c];
        Cb[ulong(r)*N + c] = @T@(y);
    }
}

// gemm_nt_bias_scatter: like gemm_nt_bias but the store epilogue writes the result
// DIRECTLY into channel-major OUT[N,Cout,(Dout,)Hout,Wout], removing the out_flat[P,Cout]
// staging buffer + the .permute().contiguous() copy. Unifies 2D/3D: for 2D pass Dout=1.
// PRM: [M, N, K, has_bias, p0, Cout, Dout, Hout, Wout]
kernel void gemm_nt_bias_scatter(
    device @T@*       A    [[buffer(0)]],
    device @T@*       Bw   [[buffer(1)]],
    device float*     BIAS [[buffer(2)]],
    device @T@*       OUT  [[buffer(3)]],
    device const int* PRM  [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    const int M=PRM[0], N=PRM[1], K=PRM[2], has_bias=PRM[3];
    const int p0=PRM[4], Cout=PRM[5], Dout=PRM[6], Hout=PRM[7], Wout=PRM[8];
    const int m0=int(tgid.x)*BM, n0=int(tgid.y)*BN;
    if (m0>=M || n0>=N) return;
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto mA = tensor<device @T@, dextents<int,2>, tensor_inline>(
                  A  + ulong(m0)*K, dextents<int,2>{K, min(BM, M-m0)}, array<int,2>{1, K});
    auto mB = tensor<device @T@, dextents<int,2>, tensor_inline>(
                  Bw + ulong(n0)*K, dextents<int,2>{K, min(BN, N-n0)}, array<int,2>{1, K});
    using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(mA)>;
    using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(mB)>;
    auto cC = op.get_destination_cooperative_tensor<AT, BT, float>();
    for (uint16_t i=0;i<cC.get_capacity();++i) if (cC.is_valid_element(i)) cC[i]=0.0f;
    op.run(mA, mB, cC);
    const int HW = Hout*Wout, DHW = Dout*HW;
    for (uint16_t i=0;i<cC.get_capacity();++i){
        if(!cC.is_valid_element(i)) continue;
        auto idx=cC.get_multidimensional_index(i);
        const int r=int(idx[1]), col=int(idx[0]);
        if(m0+r>=M || n0+col>=N) continue;
        const int pix = p0 + (m0 + r);
        const int ow = pix % Wout;
        const int oh = (pix / Wout) % Hout;
        const int od = (pix / HW) % Dout;
        const int n  = pix / DHW;
        const int c  = n0 + col;
        float y=cC[i];
        if(has_bias) y += BIAS[c];
        OUT[ ((((ulong(n)*Cout + c)*Dout + od)*Hout + oh)*Wout + ow) ] = @T@(y);
    }
}
"""

_IM2COL_3D_SRC = r"""
#include <metal_stdlib>
using namespace metal;
kernel void im2col_3d(
    device const @T@* X     [[buffer(0)]],
    device @T@*       Atile [[buffer(1)]],
    device const int* PRM   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const int Cin=PRM[1], D=PRM[2], H=PRM[3], W=PRM[4];
    const int kd=PRM[5], kh=PRM[6], kw=PRM[7];
    const int sD=PRM[8], sH=PRM[9], sW=PRM[10];
    const int pD=PRM[11], pH=PRM[12], pW=PRM[13];
    const int Dout=PRM[14], Hout=PRM[15], Wout=PRM[16];
    const int K=PRM[17], p0=PRM[18], rows=PRM[19];
    const uint total = uint(rows) * uint(K);
    if (gid >= total) return;
    const int r=int(gid)/K, kk=int(gid)%K, pix=p0+r;
    const int ow=pix%Wout, oh=(pix/Wout)%Hout, od=(pix/(Wout*Hout))%Dout;
    const int n=pix/(Wout*Hout*Dout);
    const int kj=kk%kw, ki=(kk/kw)%kh, kt=(kk/(kw*kh))%kd, c=kk/(kd*kh*kw);
    const int id_=od*sD+kt-pD, ih=oh*sH+ki-pH, iw=ow*sW+kj-pW;
    @T@ v=@T@(0);
    if (id_>=0 && id_<D && ih>=0 && ih<H && iw>=0 && iw<W)
        v = X[((((ulong(n)*Cin + c)*D + id_)*H + ih)*W + iw)];
    Atile[gid]=v;
}
"""

# combined per-dtype source is assembled in _lib(). Keep _ALL_SRC as source fragments.
_ALL_SRC = [_IM2COL_2D_SRC, _IM2COL_3D_SRC, _GEMM_SRC]


def _src(dtype_key):
    return "\n".join(_ALL_SRC).replace("@T@", dtype_key)


_libs = {}  # dtype-key -> compiled lib


def _lib(dtype):
    key = _DT[dtype]
    lib = _libs.get(key)
    if lib is None:
        lib = torch.mps.compile_shader(_src(key))
        _libs[key] = lib
    return lib


def _grid1d(total, tg=256):
    return (((total + tg - 1) // tg) * tg, 1, 1), (tg, 1, 1)


def _out_hw(H, W, kh, kw, s, p):
    Hout = (H + 2 * p[0] - kh) // s[0] + 1
    Wout = (W + 2 * p[1] - kw) // s[1] + 1
    return Hout, Wout


def _im2col_2d_tile(x, A_tile, kh, kw, s, p, Hout, Wout, p0, rows):
    N, Cin, H, W = x.shape
    K = Cin * kh * kw
    prm = torch.tensor([N, Cin, H, W, kh, kw, s[0], s[1], p[0], p[1],
                        Hout, Wout, K, p0, rows], dtype=torch.int32, device=x.device)
    total = rows * K
    threads, group = _grid1d(total)
    _lib(x.dtype).im2col_2d(x.contiguous(), A_tile, prm, threads=threads, group_size=group)


def _out_dhw(D, H, W, kd, kh, kw, s, p):
    Dout = (D + 2 * p[0] - kd) // s[0] + 1
    Hout = (H + 2 * p[1] - kh) // s[1] + 1
    Wout = (W + 2 * p[2] - kw) // s[2] + 1
    return Dout, Hout, Wout


def _im2col_3d_tile(x, A_tile, kd, kh, kw, s, p, Dout, Hout, Wout, p0, rows):
    N, Cin, D, H, W = x.shape
    K = Cin * kd * kh * kw
    prm = torch.tensor([N, Cin, D, H, W, kd, kh, kw, s[0], s[1], s[2],
                        p[0], p[1], p[2], Dout, Hout, Wout, K, p0, rows],
                       dtype=torch.int32, device=x.device)
    total = rows * K
    threads, group = _grid1d(total)
    _lib(x.dtype).im2col_3d(x.contiguous(), A_tile, prm, threads=threads, group_size=group)


def _im2col_2d_full(x, kh, kw, s, p):
    s = (s, s) if isinstance(s, int) else tuple(s)
    p = (p, p) if isinstance(p, int) else tuple(p)
    N, Cin, H, W = x.shape
    Hout, Wout = _out_hw(H, W, kh, kw, s, p)
    P, K = N * Hout * Wout, Cin * kh * kw
    A = torch.empty(P, K, device=x.device, dtype=x.dtype)
    _im2col_2d_tile(x, A, kh, kw, s, p, Hout, Wout, 0, P)
    return A


def _gemm_nt_bias(A, Bw, bias, out=None):
    """OUT[M,N] = A[M,K] @ Bw[N,K]^T + bias. Writes into `out` if given (a view into
    out_flat[p0:p0+rows]); allocates only when out is None. No per-tile alloc/copy."""
    M, K = A.shape
    N = Bw.shape[0]
    if out is None:
        out = torch.empty(M, N, device=A.device, dtype=A.dtype)
    has_bias = 1 if bias is not None else 0
    bias_buf = bias.float().contiguous() if bias is not None else torch.zeros(1, device=A.device)
    prm = torch.tensor([M, N, K, has_bias], dtype=torch.int32, device=A.device)
    BM = BN = 64
    NSG = 4
    gx = (M + BM - 1) // BM
    gy = (N + BN - 1) // BN
    _lib(A.dtype).gemm_nt_bias(
        A.contiguous(), Bw.contiguous(), bias_buf, out, prm,
        threads=(gx * NSG * 32, gy, 1), group_size=(NSG * 32, 1, 1),
    )
    return out


def _scatter_on():
    # Fused channel-major scatter epilogue (drops out_flat + permute copy). Default on.
    return os.environ.get("ASFP8_CONV_SCATTER", "1").lower() in ("1", "on", "true")


def _gemm_nt_bias_scatter(A, Bw, bias, out, p0, Cout, Dout, Hout, Wout):
    """OUT[N,Cout,(Dout,)Hout,Wout] = A[rows,K] @ Bw[Cout,K]^T + bias, scattered to the
    channel-major destination by decoding pix=p0+(m0+r) -> (n,od,oh,ow). Each element is
    STORED exactly once (assign, not accumulate) and every (m<M, n<Cout) is covered across
    tiles, so `out` needs no pre-zeroing. `out` is the final channel-major tensor (no
    out_flat staging, no permute copy). For 2D pass Dout=1."""
    M, K = A.shape
    N = Bw.shape[0]
    has_bias = 1 if bias is not None else 0
    bias_buf = bias.float().contiguous() if bias is not None else torch.zeros(1, device=A.device)
    prm = torch.tensor([M, N, K, has_bias, p0, Cout, Dout, Hout, Wout],
                       dtype=torch.int32, device=A.device)
    BM = BN = 64
    NSG = 4
    gx = (M + BM - 1) // BM
    gy = (N + BN - 1) // BN
    _lib(A.dtype).gemm_nt_bias_scatter(
        A.contiguous(), Bw.contiguous(), bias_buf, out, prm,
        threads=(gx * NSG * 32, gy, 1), group_size=(NSG * 32, 1, 1),
    )
    return out


# ---------------------------------------------------------------------------
# Public conv driver (tiled) + fallback contract
# ---------------------------------------------------------------------------

def _supported(x, weight, dilation, groups, dtype):
    dil = dilation if isinstance(dilation, tuple) else (dilation,)
    return (x.device.type == "mps" and weight.device.type == "mps"
            and dtype in _DT and weight.dtype == dtype
            and groups == 1 and all(d == 1 for d in dil))


def _fallback_conv(x, weight, bias, stride, padding, dilation, groups):
    """Single named seam for the stock-conv fallback (monkeypatched by the spy test).

    MUST call the captured true originals (_orig_conv2d/_orig_conv3d) when install() has
    replaced F.conv2d/F.conv3d with our wrappers -- otherwise the fallback re-enters the
    wrapper -> conv_im2col -> (kernel raises) -> _fallback_conv -> wrapper -> ... infinite
    recursion, breaking the "never fatal" contract. Pre-install, F.conv2d IS the original."""
    if weight.dim() == 4:
        fn = _orig_conv2d if _orig_conv2d is not None else F.conv2d
    else:
        fn = _orig_conv3d if _orig_conv3d is not None else F.conv3d
    return fn(x, weight, bias, stride, padding, dilation, groups)


def conv_im2col(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Total + never-raising: unsupported -> stock conv. Safe for direct callers (BLOCKER).
    rank = weight.dim() - 2          # 2 -> conv2d, 3 -> conv3d
    if rank not in (2, 3) or not _supported(x, weight, dilation, groups, x.dtype):
        return _fallback_conv(x, weight, bias, stride, padding, dilation, groups)
    try:
        if rank == 2:
            return _conv2d_im2col_checked(x, weight, bias, stride, padding, dilation, groups)
        return _conv3d_im2col_checked(x, weight, bias, stride, padding, dilation, groups)
    except Exception as e:      # defense in depth: never raise into the caller
        print(f"{TAG} conv{rank}d kernel error, falling back ({e!r})")
        return _fallback_conv(x, weight, bias, stride, padding, dilation, groups)


def _conv2d_im2col_checked(x, weight, bias, stride, padding, dilation, groups):
    # re-validate before any output-size math; dilation!=1 / groups!=1 already excluded by
    # _supported, but assert defensively so direct callers of the checked fn can't get wrong math.
    assert groups == 1 and weight.dim() == 4
    s = (stride, stride) if isinstance(stride, int) else tuple(stride)
    p = (padding, padding) if isinstance(padding, int) else tuple(padding)
    dil = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    assert all(d == 1 for d in dil), "dilation!=1 not supported"
    N, Cin, H, W = x.shape
    Cout, _, kh, kw = weight.shape
    Hout, Wout = _out_hw(H, W, kh, kw, s, p)   # dilation==1 guaranteed -> formula needs no dilation
    P, K = N * Hout * Wout, Cin * kh * kw
    Wmat = weight.reshape(Cout, K).contiguous()
    tile_p = max(1, min(P, _TILE_BYTES // (K * x.element_size())))
    A_tile = torch.empty(tile_p, K, device=x.device, dtype=x.dtype)
    if _scatter_on():
        # Fused channel-major scatter: allocate the final output ONCE, no out_flat / copy.
        out = torch.empty(N, Cout, Hout, Wout, device=x.device, dtype=x.dtype)
        for p0 in range(0, P, tile_p):
            rows = min(tile_p, P - p0)
            view = A_tile[:rows]
            _im2col_2d_tile(x, view, kh, kw, s, p, Hout, Wout, p0, rows)
            _gemm_nt_bias_scatter(view, Wmat, bias, out, p0, Cout, 1, Hout, Wout)
        return out
    out_flat = torch.empty(P, Cout, device=x.device, dtype=x.dtype)
    for p0 in range(0, P, tile_p):
        rows = min(tile_p, P - p0)
        view = A_tile[:rows]
        _im2col_2d_tile(x, view, kh, kw, s, p, Hout, Wout, p0, rows)
        # write GEMM result DIRECTLY into the out_flat slice (no per-tile alloc/copy):
        _gemm_nt_bias(view, Wmat, bias, out=out_flat[p0:p0 + rows])
    # [P,Cout] -> [N,Hout,Wout,Cout] -> [N,Cout,Hout,Wout]
    return out_flat.reshape(N, Hout, Wout, Cout).permute(0, 3, 1, 2).contiguous()


def _conv3d_im2col_checked(x, weight, bias, stride, padding, dilation, groups):
    assert groups == 1 and weight.dim() == 5
    s = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
    p = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
    dil = (dilation,) * 3 if isinstance(dilation, int) else tuple(dilation)
    assert all(d == 1 for d in dil), "dilation!=1 not supported"
    N, Cin, D, H, W = x.shape
    Cout, _, kd, kh, kw = weight.shape
    Dout, Hout, Wout = _out_dhw(D, H, W, kd, kh, kw, s, p)
    P, K = N * Dout * Hout * Wout, Cin * kd * kh * kw
    Wmat = weight.reshape(Cout, K).contiguous()
    tile_p = max(1, min(P, _TILE_BYTES // (K * x.element_size())))
    A_tile = torch.empty(tile_p, K, device=x.device, dtype=x.dtype)
    if _scatter_on():
        # Fused channel-major scatter: allocate the final output ONCE, no out_flat / copy.
        out = torch.empty(N, Cout, Dout, Hout, Wout, device=x.device, dtype=x.dtype)
        for p0 in range(0, P, tile_p):
            rows = min(tile_p, P - p0)
            view = A_tile[:rows]
            _im2col_3d_tile(x, view, kd, kh, kw, s, p, Dout, Hout, Wout, p0, rows)
            _gemm_nt_bias_scatter(view, Wmat, bias, out, p0, Cout, Dout, Hout, Wout)
        return out
    out_flat = torch.empty(P, Cout, device=x.device, dtype=x.dtype)
    for p0 in range(0, P, tile_p):
        rows = min(tile_p, P - p0)
        view = A_tile[:rows]
        _im2col_3d_tile(x, view, kd, kh, kw, s, p, Dout, Hout, Wout, p0, rows)
        _gemm_nt_bias(view, Wmat, bias, out=out_flat[p0:p0 + rows])
    # [P,Cout] -> [N,Dout,Hout,Wout,Cout] -> [N,Cout,Dout,Hout,Wout]
    return out_flat.reshape(N, Dout, Hout, Wout, Cout).permute(0, 4, 1, 2, 3).contiguous()


# ---------------------------------------------------------------------------
# Guarded install (never fatal). Mirrors flash_attn_mtl.py.
# ---------------------------------------------------------------------------

# Track which ranks are installed, NOT a single bool -- so a later =3d after a =2d
# in the same process can still install conv3d (idempotence-per-mode).
_installed_ranks = set()   # subset of {2, 3}
_orig_conv2d = None
_orig_conv3d = None


_CONV_OFF = ("off", "0", "false", "none", "no")
_CONV_BOTH = ("1", "on", "true", "all", "both", "2d,3d", "3d,2d")


def _mode():
    # DEFAULT-ON for conv3d (measured ~2.7x vs stock MPS conv3d, ~31% faster SeedVR2).
    # conv2d stays OFF by default (stock conv2d is at-roofline; im2col loses there).
    # Kill switch: ASFP8_CONV_IM2COL=off. Opt conv2d in with =2d / =2d,3d (=1|on|true => both).
    return os.environ.get("ASFP8_CONV_IM2COL", "3d").strip().lower()


def _gate():
    mode = _mode()
    if mode in _CONV_OFF:
        return False
    # When the user hasn't explicitly set ASFP8_CONV_IM2COL, only default-on where the
    # Metal-4 tensor-ops matmul2d kernel actually compiles (M5 / Metal 4.1). An explicit
    # 3d/2d/1 still forces it on (the wrapper falls back per-call if a conv can't run).
    if os.environ.get("ASFP8_CONV_IM2COL") is None:
        from . import _caps
        if not _caps.has_tensor_ops_matmul2d():
            return False
    return True


def _wanted_ranks():
    m = _mode()
    if m in _CONV_OFF:
        return set()
    if m in _CONV_BOTH:
        return {2, 3}
    if m == "2d":
        return {2}
    # "3d", the default, and any unrecognized value -> conv3d only (the proven default win)
    return {3}


def _make_wrap(orig, rank):
    """Module-level so fallback tests can build the wrapper closure without MPS/install()."""
    def conv(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        try:
            if weight.dim() - 2 == rank and _supported(x, weight, dilation, groups, x.dtype):
                return conv_im2col(x, weight, bias, stride, padding, dilation, groups)
        except Exception as e:
            print(f"{TAG} conv{rank}d fell back ({e!r})")
        return orig(x, weight, bias, stride, padding, dilation, groups)
    return conv


def install():
    global _orig_conv2d, _orig_conv3d
    import sys
    if sys.platform != "darwin" or not _gate():
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    if _orig_conv2d is None:
        _orig_conv2d, _orig_conv3d = F.conv2d, F.conv3d   # capture true originals once
    want = _wanted_ranks()
    if 2 in want and 2 not in _installed_ranks:
        F.conv2d = _make_wrap(_orig_conv2d, 2)
        _installed_ranks.add(2)
    if 3 in want and 3 not in _installed_ranks:
        F.conv3d = _make_wrap(_orig_conv3d, 3)
        _installed_ranks.add(3)
    print(f"{TAG} conv im2col+matmul2d active on MPS (ranks={sorted(_installed_ranks)}, "
          f"tile={_TILE_BYTES // 1024**2}MB).")
