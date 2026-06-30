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

TAG = "[AppleSilicon-FP8/conv]"

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

# combined per-dtype source is assembled in _lib(); im2col_3d + gemm are appended in
# later tasks. Keep _ALL_SRC as the list of source fragments.
_ALL_SRC = [_IM2COL_2D_SRC]


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
