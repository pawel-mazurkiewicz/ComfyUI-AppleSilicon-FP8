r"""DECISIVE spike: native-fp8 matmul2d vs the current decode->bf16->MPS path.

The only mechanism left that could beat MPS for fp8 models is BANDWIDTH: feed fp8
weights NATIVELY into matmul2d (macOS 27 added `half x metal_fp8_e4m3_format ->
float`, SDK MPPTensorOpsMatMul2d.h lines 74-83) so the weight is read at 1 byte/elem
with NO bf16 materialization in DRAM. This spike answers, on the memory-bound regime
(small M, big weight):
  Path X (today):  Wbf16 = decode_fp8(Wfp8)   then  MPS  Ah @ Wbf16      (2 B/elem weight + decode pass)
  Path Y (probe):  fp8 matmul2d                A(half) x W(fp8) -> float  (1 B/elem weight, no decode)

Outcomes:
  - compile fails        -> fp8 matmul2d NOT reachable via compile_shader (would need
                            ObjC++ newLibraryWithSource); documented dead end for this node.
  - compiles + parity ok -> compare wall-clock + GB/s of X vs Y. Y wins only if bandwidth
                            saving beats matmul2d's compute deficit (na_gemm was 0.3-0.67x MPS).

Run (ANNOUNCE GPU use first):
  /Volumes/IMPERIAL\ SPACE/AI/ComfyUI/.venv/bin/python dev/spike_fp8_matmul2d.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _patches._common import decode_fp8

NSG = 4
BM = BN = BK = 64

_FP8_MM_MSL = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BM  = @BM@;
constant constexpr int BN  = @BN@;
constant constexpr int BK  = @BK@;
constant constexpr int NSG = @NSG@;

kernel void gemm_fp8(
    device half*  A    [[buffer(0)]],   // [M,K] half (activations)
    device uchar* Braw [[buffer(1)]],   // [K,N] fp8 e4m3 bytes (weights)
    device float* C    [[buffer(2)]],   // [M,N] f32
    device int*   SH   [[buffer(3)]],   // [M,N,K]
    uint3 tgid [[threadgroup_position_in_grid]])
{
    const int M = SH[0], N = SH[1], K = SH[2];
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;
    if (m0 >= M || n0 >= N) return;

    device metal::metal_fp8_e4m3_format* B = (device metal::metal_fp8_e4m3_format*)Braw;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto mA0 = tensor(A + ulong(m0)*K, dextents<int,2>{BK, min(BM, M - m0)}, array<int,2>{1, K});
    auto mB0 = tensor(B + n0,          dextents<int,2>{min(BN, N - n0), BK}, array<int,2>{1, N});
    using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(mA0)>;
    using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(mB0)>;
    auto cC = op.get_destination_cooperative_tensor<AT, BT, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cC.get_capacity(); ++i)
        if (cC.is_valid_element(i)) cC[i] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int kk = min(BK, K - k0);
        auto mA = tensor(A + ulong(m0)*K + k0, dextents<int,2>{kk, min(BM, M - m0)}, array<int,2>{1, K});
        auto mB = tensor(B + ulong(k0)*N + n0, dextents<int,2>{min(BN, N - n0), kk}, array<int,2>{1, N});
        op.run(mA, mB, cC);
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


def compile_kernel():
    src = (_FP8_MM_MSL.replace("@BM@", str(BM)).replace("@BN@", str(BN))
           .replace("@BK@", str(BK)).replace("@NSG@", str(NSG)))
    return torch.mps.compile_shader(src)


def fp8_matmul(lib, a_half, w_u8, M, K, N):
    c = torch.zeros(M, N, device="mps", dtype=torch.float32)
    sh = torch.tensor([M, N, K], dtype=torch.int32, device="mps")
    ntg_x = -(-M // BM); ntg_y = -(-N // BN)
    lib.gemm_fp8(a_half, w_u8, c, sh, threads=(ntg_x * NSG * 32, ntg_y, 1),
                 group_size=(NSG * 32, 1, 1))
    return c


def make_fp8_weight(K, N):
    """W[K,N] as float8_e4m3fn; return (w_u8_mps, w_fp8_cpu). Build on CPU (MPS can't
    .contiguous() fp8), move bytes via uint8 view."""
    torch.manual_seed(0)
    w_fp8_cpu = (torch.randn(K, N) * 0.3).to(torch.float8_e4m3fn).contiguous()
    w_u8 = w_fp8_cpu.view(torch.uint8).to("mps")
    return w_u8, w_fp8_cpu


def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    if not torch.backends.mps.is_available():
        print("MPS not available; abort.")
        return
    if not hasattr(torch.mps, "compile_shader"):
        print("compile_shader absent; abort.")
        return

    print("=== probe fp8 type availability in compile_shader ===")
    # The matmul2d fp8 operand type metal::metal_fp8_e4m3_format is gated behind the
    # compiler builtin __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__, which the Metal frontend
    # sets only at a recent Metal language version. torch.mps.compile_shader(source)
    # exposes no MTLCompileOptions/languageVersion knob, so we cannot raise it.
    macro_src = (r"#include <metal_stdlib>" "\n"
                 r"#if __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__" "\n"
                 r"kernel void k(uint i [[thread_position_in_grid]]) {}" "\n"
                 r"#else" "\n" r"#error FP8_TYPE_UNAVAILABLE" "\n" r"#endif" "\n")
    try:
        torch.mps.compile_shader(macro_src)
        print("  __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__ is DEFINED — fp8 types available.")
    except Exception as e:
        if "FP8_TYPE_UNAVAILABLE" in repr(e):
            print("  FINDING: __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__ is UNDEFINED at "
                  "compile_shader's Metal language version.")
            print("  compile_shader(source) has no language-version option (verified: "
                  "signature is (source: str)). Native-fp8 matmul2d is unreachable via the "
                  "pure-Python path; it would need an ObjC++ newLibraryWithSource:options: "
                  "shim setting the Metal 4 language version. Documented dead end.")
            return
        raise

    print("=== compile native-fp8 matmul2d kernel ===")
    lib = compile_kernel()
    print("  compiled OK")

    # Memory-bound (small M) + one larger shape.
    SHAPES = [(256, 3072, 3072), (256, 3072, 12288), (1024, 12288, 3072)]
    cooldown = float(os.environ.get("ASFP8_BENCH_COOLDOWN", "4.0"))

    print("\n=== parity + bench (Path X = decode->MPS, Path Y = fp8 matmul2d) ===")
    hdr = f"{'shape (M,K,N)':>22} | {'pathX ms':>9} | {'pathY ms':>9} | {'Y/X':>5} | {'parity rel':>10}"
    print(hdr); print("-" * len(hdr))
    for (M, K, N) in SHAPES:
        torch.manual_seed(1)
        a = (torch.randn(M, K) * 0.3).to(torch.bfloat16).to("mps")
        a_half = a.to(torch.half)
        w_u8, w_fp8_cpu = make_fp8_weight(K, N)
        w_fp8_mps = w_fp8_cpu.to("mps")

        # reference: half activations @ decoded-fp8 weights, fp32
        w_ref = decode_fp8(w_fp8_mps, torch.float32)
        ref = a_half.float() @ w_ref

        try:
            outY = fp8_matmul(lib, a_half, w_u8, M, K, N)
            rel = ((outY - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        except Exception as e:
            print(f"{str((M,K,N)):>22} | fp8 matmul run FAILED: {e!r}")
            continue

        # Path X (today): decode fp8->bf16 each call + MPS bf16 matmul.
        def pathX():
            wb = decode_fp8(w_fp8_mps, torch.bfloat16)
            return a.to(torch.bfloat16) @ wb
        # Path Y (probe): fp8-native matmul2d (weight stays fp8).
        def pathY():
            return fp8_matmul(lib, a_half, w_u8, M, K, N)

        tX = bench(pathX); time.sleep(cooldown)
        tY = bench(pathY); time.sleep(cooldown)
        print(f"{str((M,K,N)):>22} | {tX*1e3:9.3f} | {tY*1e3:9.3f} | {tY/tX:5.2f} | {rel:10.2e}")

    print("\n(Y/X < 1.0 means fp8-native matmul2d beats the decode->MPS path. "
          "parity rel should be ~fp8 quant noise, <5e-2.)")


if __name__ == "__main__":
    main()
