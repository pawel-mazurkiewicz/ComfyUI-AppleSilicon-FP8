"""Patch #18 (opt-in): VAE conv on MPS via tiled im2col + matmul2d tensor-op GEMM.

Routes conv2d/conv3d through `im2col_2d/3d` gather -> NT `matmul2d` GEMM (half/bf16/fp32
operands, fp32 cooperative-tensor accumulate) + fused bias, looping over output-pixel
tiles so the lowered patch buffer (`A_tile`) is capped (default 384 MB). Targets the
Session-15 SeedVR2 non-tiled conv3d decode OOM. Authored with torch.mps.compile_shader
(pure-Python authoring path, no .mm/xcrun/ninja build step).

Gating: no-op unless ASFP8_CONV_IM2COL is set (1|on|true => both ranks; 2d / 3d => that
rank only). Never fatal: any compile/kernel failure falls back to stock F.conv2d/conv3d.

B.0 probe (M5 Max / macOS 27 / PyTorch 2.11 / Metal 4.1) confirmed the <T,T,float>
cooperative-accumulate matmul2d compiles+runs+CORRECT for half, bfloat AND float, so all
three are exposed below.
"""
import os

import torch
import torch.nn.functional as F

TAG = "[AppleSilicon-FP8/conv]"

_TILE_BYTES = int(os.environ.get("ASFP8_CONV_TILE_MB", "384")) * 1024 * 1024

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
"""

# combined per-dtype source is assembled in _lib(); im2col_3d is appended in B.6.
# Keep _ALL_SRC as the list of source fragments.
_ALL_SRC = [_IM2COL_2D_SRC, _GEMM_SRC]


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


# ---------------------------------------------------------------------------
# Public conv driver (tiled) + fallback contract
# ---------------------------------------------------------------------------

def _supported(x, weight, dilation, groups, dtype):
    dil = dilation if isinstance(dilation, tuple) else (dilation,)
    return (x.device.type == "mps" and weight.device.type == "mps"
            and dtype in _DT and groups == 1 and all(d == 1 for d in dil))


def _fallback_conv(x, weight, bias, stride, padding, dilation, groups):
    """Single named seam for the stock-conv fallback (monkeypatched by the spy test)."""
    fn = F.conv2d if weight.dim() == 4 else F.conv3d
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
    out_flat = torch.empty(P, Cout, device=x.device, dtype=x.dtype)
    tile_p = max(1, min(P, _TILE_BYTES // (K * x.element_size())))
    A_tile = torch.empty(tile_p, K, device=x.device, dtype=x.dtype)
    for p0 in range(0, P, tile_p):
        rows = min(tile_p, P - p0)
        view = A_tile[:rows]
        _im2col_2d_tile(x, view, kh, kw, s, p, Hout, Wout, p0, rows)
        # write GEMM result DIRECTLY into the out_flat slice (no per-tile alloc/copy):
        _gemm_nt_bias(view, Wmat, bias, out=out_flat[p0:p0 + rows])
    # [P,Cout] -> [N,Hout,Wout,Cout] -> [N,Cout,Hout,Wout]
    return out_flat.reshape(N, Hout, Wout, Cout).permute(0, 3, 1, 2).contiguous()
