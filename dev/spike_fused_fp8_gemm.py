"""Spike: fused FP8->bf16 decode inside NA matmul2d (bandwidth probe).

Stages FP8 (uchar) A/B tiles into threadgroup bfloat via a 256-entry LUT,
then runs matmul2d on the threadgroup tensors — reads 1 byte/elem instead of 2,
so if NA is bandwidth-bound this should beat the pre-decode path.

    python dev/spike_fused_fp8_gemm.py

Emits a single JSON line on stdout (last line):
    {"correct": bool, "rel": float, "tf_fused": float, "tf_predecode": float}
"""
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from _patches._common import decode_fp8, fp8_to_float_lut
from _patches import na_gemm

_BM, _BN, _BK, _NSG = 64, 64, 64, 4

# Fused kernel: reads FP8 as uchar, decodes to threadgroup bfloat via LUT,
# then runs matmul2d on the threadgroup tiles.
_FUSED_GEMM_MSL = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BM  = @BM@;
constant constexpr int BN  = @BN@;
constant constexpr int BK  = @BK@;
constant constexpr int NSG = @NSG@;

kernel void fused_fp8_gemm(
    device uchar*  A   [[buffer(0)]],   // [M,K] FP8 as raw bytes
    device uchar*  B   [[buffer(1)]],   // [K,N] FP8 as raw bytes
    device float*  C   [[buffer(2)]],   // [M,N] output
    device bfloat* LUT [[buffer(3)]],   // [256] FP8->bf16 lookup table
    device int*    SH  [[buffer(4)]],   // [M,N,K]
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgid  [[simdgroup_index_in_threadgroup]])
{
    const int M = SH[0], N = SH[1], K = SH[2];
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;
    if (m0 >= M || n0 >= N) return;

    // Threadgroup staging buffers for decoded bf16 tiles
    threadgroup bfloat tgA[BM * BK];
    threadgroup bfloat tgB[BK * BN];

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    // Get output cooperative tensor type and zero-init accumulator
    // Use a dummy tile to infer types (same shapes as real tiles)
    auto mA0 = tensor((threadgroup bfloat*)tgA, dextents<int,2>{BK, BM}, array<int,2>{1, BK});
    auto mB0 = tensor((threadgroup bfloat*)tgB, dextents<int,2>{BN, BK}, array<int,2>{1, BN});
    using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(mA0)>;
    using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(mB0)>;
    auto cC = op.get_destination_cooperative_tensor<AT, BT, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cC.get_capacity(); ++i)
        if (cC.is_valid_element(i)) cC[i] = 0.0f;

    const int actM = min(BM, M - m0);
    const int actN = min(BN, N - n0);

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int kk = min(BK, K - k0);

        // Cooperatively decode FP8 A tile [actM, kk] -> tgA, column-major for matmul2d
        // tgA layout: [kk, actM] with stride [1, BK]
        const uint total_a = uint(actM) * uint(kk);
        for (uint idx = sgid * 32 + tiisg; idx < total_a; idx += NSG * 32) {
            const uint row = idx / uint(kk);   // m index
            const uint col = idx % uint(kk);   // k index
            uchar byte = A[(ulong(m0) + row) * K + (k0 + col)];
            tgA[col * BM + row] = LUT[byte];   // col-major: [k, m]
        }
        // Cooperatively decode FP8 B tile [kk, actN] -> tgB, column-major for matmul2d
        // tgB layout: [actN, kk] with stride [1, BN]
        const uint total_b = uint(kk) * uint(actN);
        for (uint idx = sgid * 32 + tiisg; idx < total_b; idx += NSG * 32) {
            const uint row = idx / uint(actN);  // k index
            const uint col = idx % uint(actN);  // n index
            uchar byte = B[(ulong(k0) + row) * N + (n0 + col)];
            tgB[col * BK + row] = LUT[byte];    // col-major: [n, k]
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tensor((threadgroup bfloat*)tgA, dextents<int,2>{kk, actM}, array<int,2>{1, BK});
        auto mB = tensor((threadgroup bfloat*)tgB, dextents<int,2>{actN, kk}, array<int,2>{1, BN});
        op.run(mA, mB, cC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* Cb = C + ulong(m0)*N + n0;
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cC.get_capacity(); ++i) {
        if (!cC.is_valid_element(i)) continue;
        auto idx = cC.get_multidimensional_index(i);
        const int r = int(idx[1]), c = int(idx[0]);
        if (m0 + r >= M || n0 + c >= N) continue;
        Cb[ulong(r)*N + c] = cC[i];
    }
}
"""

_fused_lib = None
_fused_compiled = None


def _get_fused_lib():
    global _fused_lib, _fused_compiled
    if _fused_compiled is not None:
        return _fused_lib if _fused_compiled else None
    if not hasattr(torch.mps, "compile_shader"):
        _fused_compiled = False
        return None
    try:
        src = (_FUSED_GEMM_MSL
               .replace("@BM@", str(_BM)).replace("@BN@", str(_BN))
               .replace("@BK@", str(_BK)).replace("@NSG@", str(_NSG)))
        _fused_lib = torch.mps.compile_shader(src)
        _fused_compiled = True
    except Exception as e:
        print(f"[spike] fused kernel did not compile: {e!r}", file=sys.stderr)
        _fused_compiled = False
    return _fused_lib


def fused_fp8_matmul(a_fp8, b_fp8, lut_bf16):
    """C[M,N] f32 = decode(A[M,K]) @ decode(B[K,N]) with inline decode."""
    lib = _get_fused_lib()
    if lib is None:
        raise RuntimeError("fused kernel unavailable")
    M, K = a_fp8.shape
    K2, N = b_fp8.shape
    assert K == K2
    # View FP8 tensors as uint8 on MPS (they're already there as bytes)
    a_u8 = a_fp8.cpu().contiguous().view(torch.uint8).to("mps")
    b_u8 = b_fp8.cpu().contiguous().view(torch.uint8).to("mps")
    c = torch.zeros(M, N, device="mps", dtype=torch.float32)
    sh = torch.tensor([M, N, K], dtype=torch.int32, device="mps")
    ntg_x = -(-M // _BM)
    ntg_y = -(-N // _BN)
    lib.fused_fp8_gemm(a_u8, b_u8, c, lut_bf16, sh,
                       threads=(ntg_x * 128, ntg_y, 1),
                       group_size=(128, 1, 1))
    return c


def bench(fn, w=5, it=30):
    for _ in range(w):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / it


def main():
    if not torch.backends.mps.is_available():
        print("MPS unavailable; skipping", file=sys.stderr)
        print(json.dumps({"correct": False, "rel": -1.0, "tf_fused": 0.0, "tf_predecode": 0.0}))
        return

    if not na_gemm.available():
        print("NA matmul2d unavailable; skipping", file=sys.stderr)
        print(json.dumps({"correct": False, "rel": -1.0, "tf_fused": 0.0, "tf_predecode": 0.0}))
        return

    lib = _get_fused_lib()
    if lib is None:
        print("Fused kernel did not compile; skipping", file=sys.stderr)
        print(json.dumps({"correct": False, "rel": -1.0, "tf_fused": 0.0, "tf_predecode": 0.0}))
        return

    # Correctness check on a small shape
    torch.manual_seed(0)
    M, K, N = 128, 256, 96
    a_fp8 = (torch.randn(M, K) * 0.3).to(torch.float8_e4m3fn)
    b_fp8 = (torch.randn(K, N) * 0.3).to(torch.float8_e4m3fn)

    lut = fp8_to_float_lut(torch.float8_e4m3fn, torch.device("mps"), torch.bfloat16)

    # Reference: pre-decode then NA matmul
    a_bf16 = decode_fp8(a_fp8, torch.bfloat16).to("mps").contiguous()
    b_bf16 = decode_fp8(b_fp8, torch.bfloat16).to("mps").contiguous()
    ref = na_gemm.na_matmul(a_bf16, b_bf16).cpu()

    # Fused path
    a_mps = a_fp8.to("mps")
    b_mps = b_fp8.to("mps")
    out = fused_fp8_matmul(a_mps, b_mps, lut).cpu()

    rel = float(((out - ref).abs().max() / (ref.abs().max() + 1e-9)).item())
    correct = rel < 5e-2
    if not correct:
        print(f"[spike] CORRECTNESS FAIL: rel={rel:.4f}", file=sys.stderr)
    else:
        print(f"[spike] correctness OK: rel={rel:.6f}", file=sys.stderr)

    # Benchmark on a FLUX-ish shape
    M, K, N = 4096, 4096, 4096
    a_fp8_b = (torch.randn(M, K) * 0.3).to(torch.float8_e4m3fn).to("mps")
    b_fp8_b = (torch.randn(K, N) * 0.3).to(torch.float8_e4m3fn).to("mps")
    a_bf16_b = decode_fp8(a_fp8_b.cpu().view(torch.uint8).view(torch.float8_e4m3fn), torch.bfloat16).to("mps").contiguous()
    b_bf16_b = decode_fp8(b_fp8_b.cpu().view(torch.uint8).view(torch.float8_e4m3fn), torch.bfloat16).to("mps").contiguous()
    flops = 2 * M * N * K

    tf_predecode = flops / bench(lambda: na_gemm.na_matmul(a_bf16_b, b_bf16_b)) / 1e12
    tf_fused = flops / bench(lambda: fused_fp8_matmul(a_fp8_b, b_fp8_b, lut)) / 1e12

    print(f"[spike] M{M} K{K} N{N}: pre-decode {tf_predecode:.1f} TF/s | fused {tf_fused:.1f} TF/s | "
          f"fused/pre-decode {tf_fused/tf_predecode:.2f}x", file=sys.stderr)

    result = {"correct": correct, "rel": round(rel, 6),
              "tf_fused": round(tf_fused, 2), "tf_predecode": round(tf_predecode, 2)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
