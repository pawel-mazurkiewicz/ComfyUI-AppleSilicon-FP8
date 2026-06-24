// fp8-native matmul2d torch MPS extension (EXPERIMENTAL probe).
//
// Compiles a Metal 4.1 kernel (half activations x metal_fp8_e4m3_format weights ->
// float) via newLibraryWithSource:options: with languageVersion 4.1 — the version
// torch.mps.compile_shader can't request, where __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__
// turns on. Dispatches on torch's own MPS stream (getCurrentMPSStream) using the
// MTLBuffers backing the input tensors (zero-copy).
//
// Kernel body mirrors the proven na_gemm matmul2d structure (cooperative tensors,
// 64x64x64 tile, 4 simdgroups), with the right operand = fp8.

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

constant constexpr int BM = 64, BN = 64, BK = 64, NSG = 4;

kernel void gemm_fp8(
    device half*  A [[buffer(0)]],                       // [M,K] half
    device uchar* B [[buffer(1)]],                       // [K,N] fp8 e4m3 bytes (1 B/elem)
    device float* C [[buffer(2)]],                       // [M,N] f32
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;
    if (m0 >= M || n0 >= N) return;

    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    using fp8_t = metal::metal_fp8_e4m3_format;
    // Full-K operands + a single op.run: let matmul2d choose its internal tileK.
    // (The manual BK-chunked accumulate loop forced tileK=BK and paid per-chunk
    // cooperative-op overhead, which dominated the compute-bound large-M regime.)
    auto mA = tensor(A + ulong(m0)*K, dextents<int,2>{K, min(BM, M - m0)}, array<int,2>{1, K});
    auto mB = tensor<device fp8_t, dextents<int,2>, tensor_inline>(
                  B + n0, dextents<int,2>{min(BN, N - n0), K}, array<int,2>{1, N});
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

// NT variant for nn.Linear: W stored [N=out, K=in] fp8; transpose_right=true so the
// op computes A[M,K] @ Wᵀ -> [M,N] reading W as stored (no fp8 transpose needed).
kernel void gemm_fp8_nt(
    device half*  A [[buffer(0)]],                       // [M,K] half
    device uchar* B [[buffer(1)]],                       // [N,K] fp8 e4m3 bytes (W as stored)
    device float* C [[buffer(2)]],                       // [M,N] f32
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;
    if (m0 >= M || n0 >= N) return;

    // NT (transpose_right=true): per the SDK, the op reads K from B.extents().extent(0),
    // so B must be described K-first: extents {K, N-tile}. W[N,K] element (k,n) = W[n0+n,k]
    // at n0*K + n*K + k -> k-stride 1, n-stride K. The op computes A @ Wᵀ -> [M,N].
    constexpr auto desc = matmul2d_descriptor(
        BM, BN, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;

    using fp8_t = metal::metal_fp8_e4m3_format;
    auto mA = tensor(A + ulong(m0)*K, dextents<int,2>{K, min(BM, M - m0)}, array<int,2>{1, K});
    auto mB = tensor<device fp8_t, dextents<int,2>, tensor_inline>(
                  B + ulong(n0)*K, dextents<int,2>{K, min(BN, N - n0)}, array<int,2>{1, K});
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
)MTL";

static id<MTLComputePipelineState> g_pso = nil;
static std::string g_compile_error;

static id<MTLComputePipelineState> get_pso() {
    if (g_pso) return g_pso;
    id<MTLDevice> dev = MPSDevice::getInstance()->device();
    MTLCompileOptions* opts = [MTLCompileOptions new];
    // Metal 4.1 is what enables __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__ (verified via xcrun).
    opts.languageVersion = MTLLanguageVersion4_1;
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kSrc];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        g_compile_error = err ? std::string(err.localizedDescription.UTF8String) : "unknown";
        TORCH_CHECK(false, "fp8 Metal library compile failed: ", g_compile_error);
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_fp8"];
    TORCH_CHECK(fn, "gemm_fp8 function not found in compiled library");
    g_pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    TORCH_CHECK(g_pso, "pipeline state creation failed: ",
                err ? err.localizedDescription.UTF8String : "unknown");
    return g_pso;
}

static id<MTLComputePipelineState> g_pso_nt = nil;

static id<MTLComputePipelineState> get_pso_nt() {
    if (g_pso_nt) return g_pso_nt;
    id<MTLDevice> dev = MPSDevice::getInstance()->device();
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion4_1;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                           options:opts error:&err];
    if (!lib) {
        g_compile_error = err ? std::string(err.localizedDescription.UTF8String) : "unknown";
        TORCH_CHECK(false, "fp8 Metal library compile failed: ", g_compile_error);
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_fp8_nt"];
    TORCH_CHECK(fn, "gemm_fp8_nt function not found in compiled library");
    g_pso_nt = [dev newComputePipelineStateWithFunction:fn error:&err];
    TORCH_CHECK(g_pso_nt, "NT pipeline state creation failed: ",
                err ? err.localizedDescription.UTF8String : "unknown");
    return g_pso_nt;
}

// C[M,N] f32 = A[M,K] half @ W[K,N] fp8(e4m3, passed as uint8 bytes).
torch::Tensor fp8_matmul2d(torch::Tensor a, torch::Tensor w_u8, int64_t N) {
    TORCH_CHECK(a.is_mps() && w_u8.is_mps(), "inputs must be on mps");
    TORCH_CHECK(a.scalar_type() == torch::kHalf, "activations must be half");
    TORCH_CHECK(a.is_contiguous() && w_u8.is_contiguous(), "inputs must be contiguous");
    const int M = (int)a.size(0), K = (int)a.size(1);
    auto C = torch::zeros({(long)M, (long)N}, a.options().dtype(torch::kFloat32));

    id<MTLComputePipelineState> pso = get_pso();
    MPSStream* stream = getCurrentMPSStream();

    id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a.storage().data());
    id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, w_u8.storage().data());
    id<MTLBuffer> cBuf = __builtin_bit_cast(id<MTLBuffer>, C.storage().data());
    const NSUInteger aOff = a.storage_offset() * a.element_size();
    const NSUInteger bOff = w_u8.storage_offset() * w_u8.element_size();
    const NSUInteger cOff = C.storage_offset() * C.element_size();

    int Mi = M, Ni = (int)N, Ki = K;
    const int BM = 64, BN = 64, NSG = 4;
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
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                threadsPerThreadgroup:MTLSizeMake(NSG * 32, 1, 1)];
        }
    });
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
    return C;
}

// C[M,N] f32 = A[M,K] half @ Wᵀ where W is [N,K] fp8 (e4m3 bytes). For nn.Linear.
torch::Tensor fp8_matmul2d_nt(torch::Tensor a, torch::Tensor w_u8, int64_t N) {
    TORCH_CHECK(a.is_mps() && w_u8.is_mps(), "inputs must be on mps");
    TORCH_CHECK(a.scalar_type() == torch::kHalf, "activations must be half");
    TORCH_CHECK(a.is_contiguous() && w_u8.is_contiguous(), "inputs must be contiguous");
    const int M = (int)a.size(0), K = (int)a.size(1);
    auto C = torch::zeros({(long)M, (long)N}, a.options().dtype(torch::kFloat32));

    id<MTLComputePipelineState> pso = get_pso_nt();
    MPSStream* stream = getCurrentMPSStream();

    id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a.storage().data());
    id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, w_u8.storage().data());
    id<MTLBuffer> cBuf = __builtin_bit_cast(id<MTLBuffer>, C.storage().data());
    const NSUInteger aOff = a.storage_offset() * a.element_size();
    const NSUInteger bOff = w_u8.storage_offset() * w_u8.element_size();
    const NSUInteger cOff = C.storage_offset() * C.element_size();

    int Mi = M, Ni = (int)N, Ki = K;
    const int BM = 64, BN = 64, NSG = 4;
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
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                threadsPerThreadgroup:MTLSizeMake(NSG * 32, 1, 1)];
        }
    });
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_matmul2d", &fp8_matmul2d, "fp8 e4m3 matmul2d (half x fp8[K,N] -> f32) on MPS");
    m.def("fp8_matmul2d_nt", &fp8_matmul2d_nt, "NT fp8 matmul (half x Wᵀ, W=[N,K] fp8) on MPS");
    m.def("warmup", []() { get_pso(); get_pso_nt(); return true; }, "compile + build pipelines");
}
