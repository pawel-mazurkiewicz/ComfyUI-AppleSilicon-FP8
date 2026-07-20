// int4b-weight matmul2d torch MPS extension (EXPERIMENTAL go/no-go probe).
//
// Mirrors int8_ext/int8_matmul2d.mm, but the B (weight) operand is PACKED int4
// (`metal::int4b_format`, 2 values/byte) — the W4 tier of the M5 Neural
// Accelerators. Two questions this answers:
//   1. correctness: does MPP's int4b nibble order match comfy-kitchen's
//      row-major packing (low nibble = even column)?
//   2. speed: what do int8xint4b->int32 (W4A8) and bfloatxint4b->float (W4A16)
//      reach at Krea2/FLUX shapes vs our int8 W8A8 kernel and MPS bf16 GEMM?
// Weights stay packed in device memory — half the weight bandwidth of int8.
//
// NT layout (transpose_right=true) to match nn.Linear: A[M,K] @ Wᵀ, W=[N,K].

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

using namespace at::mps;

static const char* kSrc = R"MTL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#if !defined(__HAVE_INT4B_FORMAT_TYPE__)
#error "int4b_format unavailable at this Metal language version"
#endif

// Tile config from the dev/probe_int4_tune.py sweep (12 configs, all verified
// bit-exact): BM128/BN128/NSG8 dv_i4 = 86.9 TF/s vs BM64/BN64/NSG4's 80.3 TF/s.
// Bigger is NOT better here — BM256 collapses to 20.8 TF/s (cooperative-tensor
// register spill), and tg_i4 threadgroup staging is 2-4x worse at every config
// (B is already read once per m-block, so staging adds copies and barriers
// without removing traffic).
constant constexpr int BM = 128, BN = 128, NSG = 8;

// C[M,N] int32 = A[M,K] int8 @ Wᵀ where W is [N,K] packed int4 (K/2 bytes/row).
kernel void gemm_i8i4_nt(
    device signed char* A [[buffer(0)]],
    device uchar*       B [[buffer(1)]],   // [N, K/2] bytes, int4b packed
    device int*         C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;
    if (m0 >= M || n0 >= N) return;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto mA = tensor<device signed char, dextents<int,2>, tensor_inline>(
                  A + ulong(m0)*K, dextents<int,2>{K, min(BM, M - m0)}, array<int,2>{1, K});
    auto mB = tensor<device int4b_format, dextents<int,2>, tensor_inline>(
                  B + ulong(n0)*(K/2), dextents<int,2>{K, min(BN, N - n0)}, array<int,2>{1, K});
    using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(mA)>;
    using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(mB)>;
    auto cC = op.get_destination_cooperative_tensor<AT, BT, int>();
    for (uint16_t i = 0; i < cC.get_capacity(); ++i)
        if (cC.is_valid_element(i)) cC[i] = 0;

    op.run(mA, mB, cC);

    device int* Cb = C + ulong(m0)*N + n0;
    for (uint16_t i = 0; i < cC.get_capacity(); ++i) {
        if (!cC.is_valid_element(i)) continue;
        auto idx = cC.get_multidimensional_index(i);
        const int r = int(idx[1]), c = int(idx[0]);
        if (m0 + r >= M || n0 + c >= N) continue;
        Cb[ulong(r)*N + c] = cC[i];
    }
}

// C[M,N] float = A[M,K] bfloat @ Wᵀ where W is [N,K] packed int4.
kernel void gemm_bf16i4_nt(
    device bfloat* A [[buffer(0)]],
    device uchar*  B [[buffer(1)]],
    device float*  C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;
    if (m0 >= M || n0 >= N) return;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto mA = tensor<device bfloat, dextents<int,2>, tensor_inline>(
                  A + ulong(m0)*K, dextents<int,2>{K, min(BM, M - m0)}, array<int,2>{1, K});
    auto mB = tensor<device int4b_format, dextents<int,2>, tensor_inline>(
                  B + ulong(n0)*(K/2), dextents<int,2>{K, min(BN, N - n0)}, array<int,2>{1, K});
    using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(mA)>;
    using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(mB)>;
    auto cC = op.get_destination_cooperative_tensor<AT, BT, float>();
    for (uint16_t i = 0; i < cC.get_capacity(); ++i)
        if (cC.is_valid_element(i)) cC[i] = 0.0f;

    op.run(mA, mB, cC);

    device float* Cb = C + ulong(m0)*N + n0;
    for (uint16_t i = 0; i < cC.get_capacity(); ++i) {
        if (!cC.is_valid_element(i)) continue;
        auto idx = cC.get_multidimensional_index(i);
        const int r = int(idx[1]), c = int(idx[0]);
        if (m0 + r >= M || n0 + c >= N) continue;
        Cb[ulong(r)*N + c] = cC[i];
    }
}
// C[M,N] bf16 = dequant(A_int8[M,K] @ Wᵀ_int4b[N,K]) fused epilogue:
//   C[m,n] = bf16(float(acc) * x_scale[m] * w_scale[n]) (+ bias[n] in bf16)
// Mirrors int8_gemm.mm's store-epilogue contract: int32→fp32, fp32 rescale,
// round to bf16, then a separate bf16-precision bias add.
kernel void gemm_i8i4_fused_nt(
    device signed char* A [[buffer(0)]],
    device uchar*       B [[buffer(1)]],
    device bfloat*      C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    device float* x_scale [[buffer(6)]],   // [M] per-row activation scale
    device float* w_scale [[buffer(7)]],   // [N] per-row weight scale
    device bfloat* bias [[buffer(8)]],     // [N] or unused
    constant int& has_bias [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;
    if (m0 >= M || n0 >= N) return;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    auto mA = tensor<device signed char, dextents<int,2>, tensor_inline>(
                  A + ulong(m0)*K, dextents<int,2>{K, min(BM, M - m0)}, array<int,2>{1, K});
    auto mB = tensor<device int4b_format, dextents<int,2>, tensor_inline>(
                  B + ulong(n0)*(K/2), dextents<int,2>{K, min(BN, N - n0)}, array<int,2>{1, K});
    using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(mA)>;
    using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(mB)>;
    auto cC = op.get_destination_cooperative_tensor<AT, BT, int>();
    for (uint16_t i = 0; i < cC.get_capacity(); ++i)
        if (cC.is_valid_element(i)) cC[i] = 0;

    op.run(mA, mB, cC);

    device bfloat* Cb = C + ulong(m0)*N + n0;
    for (uint16_t i = 0; i < cC.get_capacity(); ++i) {
        if (!cC.is_valid_element(i)) continue;
        auto idx = cC.get_multidimensional_index(i);
        const int r = int(idx[1]), c = int(idx[0]);
        if (m0 + r >= M || n0 + c >= N) continue;
        float acc = float(cC[i]) * x_scale[m0 + r] * w_scale[n0 + c];
        bfloat v = bfloat(acc);
        if (has_bias) v = v + bias[n0 + c];
        Cb[ulong(r)*N + c] = v;
    }
}
)MTL";

static id<MTLLibrary> g_lib = nil;
static id<MTLComputePipelineState> g_pso_i8i4 = nil;
static id<MTLComputePipelineState> g_pso_bf16i4 = nil;
static id<MTLComputePipelineState> g_pso_i8i4_fused = nil;

static id<MTLLibrary> get_lib() {
    if (g_lib) return g_lib;
    id<MTLDevice> dev = MPSDevice::getInstance()->device();
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion4_1;
    NSError* err = nil;
    g_lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                              options:opts error:&err];
    TORCH_CHECK(g_lib, "int4 Metal library compile failed: ",
                err ? err.localizedDescription.UTF8String : "unknown");
    return g_lib;
}

static id<MTLComputePipelineState> get_pso(const char* name,
                                           id<MTLComputePipelineState>* slot) {
    if (*slot) return *slot;
    id<MTLDevice> dev = MPSDevice::getInstance()->device();
    NSError* err = nil;
    id<MTLFunction> fn = [get_lib() newFunctionWithName:[NSString stringWithUTF8String:name]];
    TORCH_CHECK(fn, name, " not found in compiled library");
    *slot = [dev newComputePipelineStateWithFunction:fn error:&err];
    TORCH_CHECK(*slot, name, " pipeline state creation failed: ",
                err ? err.localizedDescription.UTF8String : "unknown");
    return *slot;
}

static void dispatch_nt(id<MTLComputePipelineState> pso,
                        torch::Tensor a, torch::Tensor w4, torch::Tensor C,
                        int M, int N, int K) {
    MPSStream* stream = getCurrentMPSStream();
    id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a.storage().data());
    id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, w4.storage().data());
    id<MTLBuffer> cBuf = __builtin_bit_cast(id<MTLBuffer>, C.storage().data());
    const NSUInteger aOff = a.storage_offset() * a.element_size();
    const NSUInteger bOff = w4.storage_offset() * w4.element_size();
    const NSUInteger cOff = C.storage_offset() * C.element_size();
    const int BM = 128, BN = 128, NSG = 8;  // must match kSrc
    const NSUInteger gx = (M + BM - 1) / BM, gy = (N + BN - 1) / BN;
    dispatch_sync(stream->queue(), ^(){
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
            [enc setComputePipelineState:pso];
            [enc setBuffer:aBuf offset:aOff atIndex:0];
            [enc setBuffer:bBuf offset:bOff atIndex:1];
            [enc setBuffer:cBuf offset:cOff atIndex:2];
            [enc setBytes:&M length:sizeof(int) atIndex:3];
            [enc setBytes:&N length:sizeof(int) atIndex:4];
            [enc setBytes:&K length:sizeof(int) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                threadsPerThreadgroup:MTLSizeMake(NSG * 32, 1, 1)];
        }
    });
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
}

// C[M,N] int32 = A[M,K] int8 @ Wᵀ, W packed int4 [N, K/2] uint8.
torch::Tensor i8i4_matmul2d_nt(torch::Tensor a_i8, torch::Tensor w4, int64_t K, int64_t N) {
    TORCH_CHECK(a_i8.is_mps() && w4.is_mps(), "inputs must be on mps");
    TORCH_CHECK(a_i8.scalar_type() == torch::kChar, "A must be int8");
    TORCH_CHECK(w4.scalar_type() == torch::kByte || w4.scalar_type() == torch::kChar,
                "W must be packed int4 bytes");
    TORCH_CHECK(a_i8.is_contiguous() && w4.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(K % 2 == 0, "K must be even");
    const int M = (int)a_i8.size(0);
    auto C = torch::zeros({(long)M, (long)N}, a_i8.options().dtype(torch::kInt32));
    dispatch_nt(get_pso("gemm_i8i4_nt", &g_pso_i8i4), a_i8, w4, C, M, (int)N, (int)K);
    return C;
}

// C[M,N] float = A[M,K] bfloat @ Wᵀ, W packed int4 [N, K/2] uint8.
torch::Tensor bf16i4_matmul2d_nt(torch::Tensor a_bf, torch::Tensor w4, int64_t K, int64_t N) {
    TORCH_CHECK(a_bf.is_mps() && w4.is_mps(), "inputs must be on mps");
    TORCH_CHECK(a_bf.scalar_type() == torch::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(w4.scalar_type() == torch::kByte || w4.scalar_type() == torch::kChar,
                "W must be packed int4 bytes");
    TORCH_CHECK(a_bf.is_contiguous() && w4.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(K % 2 == 0, "K must be even");
    const int M = (int)a_bf.size(0);
    auto C = torch::zeros({(long)M, (long)N}, a_bf.options().dtype(torch::kFloat32));
    dispatch_nt(get_pso("gemm_bf16i4_nt", &g_pso_bf16i4), a_bf, w4, C, M, (int)N, (int)K);
    return C;
}

// C[M,N] bf16 = (A_int8 @ Wᵀ_int4b) * x_scale[m] * w_scale[n] (+ bias), fused.
torch::Tensor i8i4_linear_fused_nt(torch::Tensor a_i8, torch::Tensor w4,
                                   torch::Tensor x_scale, torch::Tensor w_scale,
                                   c10::optional<torch::Tensor> bias,
                                   int64_t K, int64_t N) {
    TORCH_CHECK(a_i8.is_mps() && w4.is_mps(), "inputs must be on mps");
    TORCH_CHECK(a_i8.scalar_type() == torch::kChar, "A must be int8");
    TORCH_CHECK(w4.scalar_type() == torch::kByte || w4.scalar_type() == torch::kChar,
                "W must be packed int4 bytes");
    TORCH_CHECK(x_scale.scalar_type() == torch::kFloat && w_scale.scalar_type() == torch::kFloat,
                "scales must be float32");
    TORCH_CHECK(a_i8.is_contiguous() && w4.is_contiguous() &&
                x_scale.is_contiguous() && w_scale.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(K % 2 == 0, "K must be even");
    const int M = (int)a_i8.size(0);
    TORCH_CHECK(x_scale.numel() == M && w_scale.numel() == N, "scale shapes");
    torch::Tensor b;
    const int has_bias = bias.has_value() ? 1 : 0;
    if (has_bias) {
        b = bias.value();
        TORCH_CHECK(b.is_mps() && b.scalar_type() == torch::kBFloat16 && b.is_contiguous(),
                    "bias must be contiguous bf16 on mps");
        TORCH_CHECK(b.numel() == N, "bias shape");
    } else {
        b = w_scale;  // placeholder binding, never read (has_bias=0)
    }
    auto C = torch::empty({(long)M, (long)N}, a_i8.options().dtype(torch::kBFloat16));

    id<MTLComputePipelineState> pso = get_pso("gemm_i8i4_fused_nt", &g_pso_i8i4_fused);
    MPSStream* stream = getCurrentMPSStream();
    id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a_i8.storage().data());
    id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, w4.storage().data());
    id<MTLBuffer> cBuf = __builtin_bit_cast(id<MTLBuffer>, C.storage().data());
    id<MTLBuffer> xsBuf = __builtin_bit_cast(id<MTLBuffer>, x_scale.storage().data());
    id<MTLBuffer> wsBuf = __builtin_bit_cast(id<MTLBuffer>, w_scale.storage().data());
    id<MTLBuffer> biBuf = __builtin_bit_cast(id<MTLBuffer>, b.storage().data());
    const NSUInteger aOff = a_i8.storage_offset() * a_i8.element_size();
    const NSUInteger bOff = w4.storage_offset() * w4.element_size();
    const NSUInteger cOff = C.storage_offset() * C.element_size();
    const NSUInteger xsOff = x_scale.storage_offset() * x_scale.element_size();
    const NSUInteger wsOff = w_scale.storage_offset() * w_scale.element_size();
    const NSUInteger biOff = b.storage_offset() * b.element_size();
    int Mi = M, Ni = (int)N, Ki = (int)K, hb = has_bias;
    const int BM = 128, BN = 128, NSG = 8;  // must match kSrc
    const NSUInteger gx = (M + BM - 1) / BM, gy = (N + BN - 1) / BN;
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
            [enc setBuffer:xsBuf offset:xsOff atIndex:6];
            [enc setBuffer:wsBuf offset:wsOff atIndex:7];
            [enc setBuffer:biBuf offset:biOff atIndex:8];
            [enc setBytes:&hb length:sizeof(int) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                threadsPerThreadgroup:MTLSizeMake(NSG * 32, 1, 1)];
        }
    });
    // Do NOT COMMIT_AND_WAIT here (mirrors int8_gemm.mm): the kernel is encoded on
    // torch's MPS stream, so ordering is preserved on the same command buffer and torch
    // commits it naturally. A per-Linear commit+wait serializes the pipeline and kills
    // async overlap (measured slowdown on int8). The result syncs when the caller reads C.
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("i8i4_linear_fused_nt", &i8i4_linear_fused_nt,
          "NT W4A8 linear with fused per-row dequant + bias (int8 x packed-int4 -> bf16) on MPS");
    m.def("i8i4_matmul2d_nt", &i8i4_matmul2d_nt, "NT W4A8 matmul (int8 x packed-int4 -> int32) on MPS");
    m.def("bf16i4_matmul2d_nt", &bf16i4_matmul2d_nt, "NT W4A16 matmul (bf16 x packed-int4 -> float) on MPS");
    m.def("warmup", []() { get_lib(); return true; }, "compile library");
}
