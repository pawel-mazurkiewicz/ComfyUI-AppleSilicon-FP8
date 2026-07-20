// int4b matmul2d TUNING harness — sweep BM/BN/NSG/BK and dv_i4 vs tg_i4.
//
// Why: int4_matmul2d.mm was written as a go/no-go correctness probe (BM=64,
// BN=64, NSG=4, one op.run over the whole K straight from device memory). It was
// never tuned, so benchmarking it against the register-tiled int8_gemm.mm and
// concluding "M5 doesn't accelerate int4" was invalid.
//
// Header fact (MPPTensorOpsMatMul2dImpl.h): int4b_format has NO cooperative-input
// intrinsics — every variant is dv_i4 (device) or tg_i4 (threadgroup). So the
// register-tiling int8 uses is unavailable for int4; the only levers are tile
// shape and staging the packed weight through threadgroup memory (tg_i4).
//
// Builds one pipeline per config, cached. Correctness is checked in Python
// against the int32 reference for every config before it is timed.

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <map>
#include <string>

using namespace at::mps;

static const char* kSrcTemplate = R"MTL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#if !defined(__HAVE_INT4B_FORMAT_TYPE__)
#error "int4b_format unavailable at this Metal language version"
#endif

constant constexpr int BM = @BM@, BN = @BN@, NSG = @NSG@, BK = @BK@;
#define USE_TG @USE_TG@

kernel void gemm_tuned(
    device signed char* A [[buffer(0)]],
    device uchar*       B [[buffer(1)]],
    device int*         C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;
    if (m0 >= M || n0 >= N) return;
    const int rows = min(BM, M - m0);
    const int cols = min(BN, N - n0);
    device int* Cb = C + ulong(m0)*N + n0;

#if USE_TG
    // Stage the packed weight tile [BN, BK/2] bytes into threadgroup memory and
    // accumulate over K. B is read from device once per (m-block, k-block).
    threadgroup uchar sB[BN * BK / 2];

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, BK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto mA0 = tensor<device signed char, dextents<int,2>, tensor_inline>(
                   A + ulong(m0)*K, dextents<int,2>{BK, rows}, array<int,2>{1, K});
    auto tB0 = tensor<threadgroup int4b_format, dextents<int,2>, tensor_inline>(
                   sB, dextents<int,2>{BK, BN}, array<int,2>{1, BK});
    using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(mA0)>;
    using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(tB0)>;
    auto cC = op.get_destination_cooperative_tensor<AT, BT, int>();
    for (uint16_t i = 0; i < cC.get_capacity(); ++i)
        if (cC.is_valid_element(i)) cC[i] = 0;

    const uint nthreads = NSG * 32u;
    const int halfBK = BK / 2, halfK = K / 2;
    for (int k0 = 0; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < uint(BN * halfBK); i += nthreads) {
            const int n  = int(i) / halfBK;
            const int kb = int(i) % halfBK;
            const int ko = k0 / 2 + kb;
            uchar v = 0;
            if (n < cols && ko < halfK) v = B[ulong(n0 + n)*halfK + ko];
            sB[i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto mA = tensor<device signed char, dextents<int,2>, tensor_inline>(
                      A + ulong(m0)*K + k0, dextents<int,2>{BK, rows}, array<int,2>{1, K});
        auto tB = tensor<threadgroup int4b_format, dextents<int,2>, tensor_inline>(
                      sB, dextents<int,2>{BK, BN}, array<int,2>{1, BK});
        op.run(mA, tB, cC);
    }

    for (uint16_t i = 0; i < cC.get_capacity(); ++i) {
        if (!cC.is_valid_element(i)) continue;
        auto idx = cC.get_multidimensional_index(i);
        const int r = int(idx[1]), c = int(idx[0]);
        if (r >= rows || c >= cols) continue;
        Cb[ulong(r)*N + c] = cC[i];
    }
#else
    // Baseline: one op.run over the whole K, weight read straight from device.
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto mA = tensor<device signed char, dextents<int,2>, tensor_inline>(
                  A + ulong(m0)*K, dextents<int,2>{K, rows}, array<int,2>{1, K});
    auto mB = tensor<device int4b_format, dextents<int,2>, tensor_inline>(
                  B + ulong(n0)*(K/2), dextents<int,2>{K, cols}, array<int,2>{1, K});
    using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(mA)>;
    using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(mB)>;
    auto cC = op.get_destination_cooperative_tensor<AT, BT, int>();
    for (uint16_t i = 0; i < cC.get_capacity(); ++i)
        if (cC.is_valid_element(i)) cC[i] = 0;

    op.run(mA, mB, cC);

    for (uint16_t i = 0; i < cC.get_capacity(); ++i) {
        if (!cC.is_valid_element(i)) continue;
        auto idx = cC.get_multidimensional_index(i);
        const int r = int(idx[1]), c = int(idx[0]);
        if (r >= rows || c >= cols) continue;
        Cb[ulong(r)*N + c] = cC[i];
    }
#endif
}
)MTL";

static std::string substitute(int BM, int BN, int NSG, int BK, int use_tg) {
    std::string s(kSrcTemplate);
    auto rep = [&s](const std::string& k, int v) {
        std::string val = std::to_string(v);
        size_t p;
        while ((p = s.find(k)) != std::string::npos) s.replace(p, k.size(), val);
    };
    rep("@BM@", BM); rep("@BN@", BN); rep("@NSG@", NSG);
    rep("@BK@", BK); rep("@USE_TG@", use_tg);
    return s;
}

static std::map<std::string, id<MTLComputePipelineState>> g_cache;

static id<MTLComputePipelineState> get_pso(int BM, int BN, int NSG, int BK, int use_tg) {
    char key[128];
    snprintf(key, sizeof(key), "%d_%d_%d_%d_%d", BM, BN, NSG, BK, use_tg);
    auto it = g_cache.find(key);
    if (it != g_cache.end()) return it->second;

    id<MTLDevice> dev = MPSDevice::getInstance()->device();
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion4_1;
    NSError* err = nil;
    std::string src = substitute(BM, BN, NSG, BK, use_tg);
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()]
                                           options:opts error:&err];
    TORCH_CHECK(lib, "int4 tune compile failed (", key, "): ",
                err ? err.localizedDescription.UTF8String : "unknown");
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_tuned"];
    TORCH_CHECK(fn, "gemm_tuned not found (", key, ")");
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    TORCH_CHECK(pso, "int4 tune pipeline failed (", key, "): ",
                err ? err.localizedDescription.UTF8String : "unknown");
    g_cache[key] = pso;
    return pso;
}

torch::Tensor i8i4_tuned(torch::Tensor a_i8, torch::Tensor w4, int64_t K, int64_t N,
                         int64_t BM, int64_t BN, int64_t NSG, int64_t BK, int64_t use_tg) {
    TORCH_CHECK(a_i8.is_mps() && w4.is_mps(), "inputs must be on mps");
    TORCH_CHECK(a_i8.scalar_type() == torch::kChar, "A must be int8");
    TORCH_CHECK(a_i8.is_contiguous() && w4.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(K % 2 == 0, "K must be even");
    if (use_tg) TORCH_CHECK(K % BK == 0, "K must be divisible by BK for tg staging");

    const int M = (int)a_i8.size(0);
    auto C = torch::zeros({(long)M, (long)N}, a_i8.options().dtype(torch::kInt32));

    id<MTLComputePipelineState> pso = get_pso((int)BM, (int)BN, (int)NSG, (int)BK, (int)use_tg);

    MPSStream* stream = getCurrentMPSStream();
    id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a_i8.storage().data());
    id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, w4.storage().data());
    id<MTLBuffer> cBuf = __builtin_bit_cast(id<MTLBuffer>, C.storage().data());
    const NSUInteger aOff = a_i8.storage_offset() * a_i8.element_size();
    const NSUInteger bOff = w4.storage_offset() * w4.element_size();
    const NSUInteger cOff = C.storage_offset() * C.element_size();
    const int Mi = M, Ni = (int)N, Ki = (int)K;
    const NSUInteger gx = (M + BM - 1) / BM, gy = (N + BN - 1) / BN;
    const NSUInteger tpg = NSG * 32;

    dispatch_sync(stream->queue(), ^(){
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
            [enc setComputePipelineState:pso];
            [enc setBuffer:aBuf offset:aOff atIndex:0];
            [enc setBuffer:bBuf offset:bOff atIndex:1];
            [enc setBuffer:cBuf offset:cOff atIndex:2];
            [enc setBytes:&Mi length:sizeof(int) atIndex:3];
            [enc setBytes:&Ni length:sizeof(int) atIndex:4];
            [enc setBytes:&Ki length:sizeof(int) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        }
    });
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("i8i4_tuned", &i8i4_tuned,
        "Tunable NT int8 x packed-int4 matmul: C[M,N] int32 = A[M,K] @ B[N,K]^T",
        pybind11::arg("a_i8"), pybind11::arg("w4"), pybind11::arg("K"), pybind11::arg("N"),
        pybind11::arg("BM"), pybind11::arg("BN"), pybind11::arg("NSG"),
        pybind11::arg("BK"), pybind11::arg("use_tg"));
}
