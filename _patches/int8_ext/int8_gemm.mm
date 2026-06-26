// ============================================================
// Bit-exact INT8×INT8→INT32 TensorOps GEMM for torch / MPS (M5+, Metal 4)
//
// C[M,N] (int32) = A[M,K] (int8) @ B[N,K]^T (int8)   -- NT layout
//
// This is the matmul our S14 probe could not make fast: the naive
// "single op.run over the whole block" cooperative-tensor kernel
// plateaued at ~75 TF/s (fp16-class). The structure here instead does
// CUTLASS-style manual register tiling — small 16x32x16 cooperative
// fragments, int32 accumulators held in registers, explicit BK=512/SK=32
// K-blocking across 16 simdgroups (WM=4 x WN=4) — which saturates the
// int8 units (~1.8x over bf16 at GEMM shapes).
//
// The Metal kernel body (helpers + w8a8_gemm_int32_impl + entry points)
// is ported from Cider (https://github.com/Mininglamp-AI/cider,
// cider/kernels/w8a8_matmul.metal), MIT License, Copyright (c) 2026
// Mininglamp contributors. Only the host dispatch is ours (torch MPS
// ObjC++ shim mirroring _patches/fp8_ext/fp8_matmul2d.mm).
// ============================================================

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

// ── NAXFrag layout constants ────────────────────────────────────
constant constexpr short kElemsPerFrag = 8;
constant constexpr short kElemCols = 4;
constant constexpr short kElemRowsJump = 8;

// ── NAXFrag coordinate mapping ──────────────────────────────────
inline short2 nax_get_coord(ushort lid) {
  short qid = short(lid >> 2);
  short fm = ((qid & 4) | ((short(lid) >> 1) & 3));
  short fn = ((qid & 2) | (short(lid) & 1)) * 4;
  return short2{fn, fm};
}

// ── Fragment load: device → register ────────────────────────────
template <typename T>
inline void nax_frag_load(thread T *dst, const device T *src, int ld, short2 sc,
                          short off_m = 0, short off_n = 0) {
  src += (sc.y + off_m) * ld + (sc.x + off_n);
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      dst[i * kElemCols + j] = src[(i * kElemRowsJump) * ld + j];
    }
  }
}

// ── Fragment store: raw INT32 (no dequant) ───────────────────
inline void nax_frag_store_int32(const thread int32_t *src, device int32_t *dst,
                                 int ld, short2 sc, short off_m, short off_n,
                                 uint M, uint N, uint m_base, uint n_base) {
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      uint ni = n_base + sc.x + off_n + j;
      if (mi < M && ni < N) {
        dst[(sc.y + off_m + i * kElemRowsJump) * ld + (sc.x + off_n + j)] =
            src[i * kElemCols + j];
      }
    }
  }
}

// ── Raw INT32 GEMM impl (B is [N, K], transpose_b=true) ─────────
// Computes: C[M,N] = A[M,K] × B[N,K]^T
template <int BM, int BN, int BK, int SK, int WM, int WN>
void w8a8_gemm_int32_impl(const device int8_t *A, const device int8_t *B,
                          device int32_t *C, uint M, uint N, uint K,
                          uint swizzle_log, uint tiles_m, uint tiles_n,
                          uint2 tgid, uint sgid, uint lid) {
  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  uint tid_y = (tgid.y << swizzle_log) + (tgid.x & ((1u << swizzle_log) - 1u));
  uint tid_x = tgid.x >> swizzle_log;
  if (tid_x >= tiles_n || tid_y >= tiles_m) {
    return;
  }

  short2 sc = nax_get_coord(ushort(lid));
  uint sg_row = sgid / WN;
  uint sg_col = sgid % WN;
  uint m_base = tid_y * BM + sg_row * SM;
  uint n_base = tid_x * BN + sg_col * SN;

  const device int8_t *sg_A = A + m_base * K;
  const device int8_t *sg_B = B + n_base * K;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  auto ct_a =
      gemm_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      gemm_op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_c =
      gemm_op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b),
                                                 int32_t>();

  int32_t c_frags[TM * TN][kElemsPerFrag];
  for (int f = 0; f < TM * TN; f++) {
    for (int i = 0; i < kElemsPerFrag; i++) {
      c_frags[f][i] = 0;
    }
  }

  int gemm_k_iters = int(K) / BK;
  for (int kk0 = 0; kk0 < gemm_k_iters; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      int8_t a_frags[TM][TK][kElemsPerFrag];
      int8_t b_frags[TN][TK][kElemsPerFrag];
      volatile int compiler_barrier;

      for (short mm = 0; mm < TM; mm++) {
        for (short kk = 0; kk < TK; kk++) {
          nax_frag_load(a_frags[mm][kk], sg_A + kk1, int(K), sc, short(mm * 16),
                        short(kk * 16));
        }
      }

      for (short nn = 0; nn < TN; nn++) {
        for (short kk = 0; kk < TK; kk++) {
          nax_frag_load(b_frags[nn][kk], sg_B + kk1, int(K), sc, short(nn * 16),
                        short(kk * 16));
        }
      }

      for (short mm = 0; mm < TM; mm++) {
        for (short nn = 0; nn < TN; nn += 2) {
          for (short kk = 0; kk < TK; kk++) {
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_a[i] = a_frags[mm][kk][i];
            }
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_b[i] = b_frags[nn][kk][i];
              ct_b[kElemsPerFrag + i] = b_frags[nn + 1][kk][i];
            }
            short c0 = mm * TN + nn, c1 = c0 + 1;
            for (short i = 0; i < kElemsPerFrag; i++) {
              ct_c[i] = c_frags[c0][i];
              ct_c[kElemsPerFrag + i] = c_frags[c1][i];
            }
            gemm_op.run(ct_a, ct_b, ct_c);
            for (short i = 0; i < kElemsPerFrag; i++) {
              c_frags[c0][i] = ct_c[i];
              c_frags[c1][i] = ct_c[kElemsPerFrag + i];
            }
          }
        }
      }
      (void)compiler_barrier;
    }
    sg_A += BK;
    sg_B += BK;
  }

  // Remainder K
  int rem_k = int(K) - gemm_k_iters * BK;
  for (int kk1 = 0; kk1 < rem_k; kk1 += 16) {
    int8_t a_frag[TM][kElemsPerFrag];
    int8_t b_frag[TN][kElemsPerFrag];
    short psk = short(max(0, rem_k - kk1));

    for (short mm = 0; mm < TM; mm++) {
      const device int8_t *ptr = sg_A + kk1 + (sc.y + mm * 16) * K + sc.x;
      for (short i = 0; i < 2; i++)
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.x + j);
          a_frag[mm][i * kElemCols + j] =
              (ki < psk) ? ptr[(i * kElemRowsJump) * K + j] : int8_t(0);
        }
    }
    for (short nn = 0; nn < TN; nn++) {
      const device int8_t *ptr = sg_B + kk1 + (sc.y + nn * 16) * K + sc.x;
      for (short i = 0; i < 2; i++) {
        for (short j = 0; j < kElemCols; j++) {
          short ki = short(sc.x + j);
          b_frag[nn][i * kElemCols + j] =
              (ki < psk) ? ptr[(i * kElemRowsJump) * K + j] : int8_t(0);
        }
      }
    }
    for (short mm = 0; mm < TM; mm++) {
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_a[i] = a_frag[mm][i];
      }
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_b[i] = b_frag[0][i];
        ct_b[kElemsPerFrag + i] = b_frag[1][i];
      }
      short c0 = mm * TN, c1 = c0 + 1;
      for (short i = 0; i < kElemsPerFrag; i++) {
        ct_c[i] = c_frags[c0][i];
        ct_c[kElemsPerFrag + i] = c_frags[c1][i];
      }
      gemm_op.run(ct_a, ct_b, ct_c);
      for (short i = 0; i < kElemsPerFrag; i++) {
        c_frags[c0][i] = ct_c[i];
        c_frags[c1][i] = ct_c[kElemsPerFrag + i];
      }
    }
  }

  // Store raw INT32
  device int32_t *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++) {
    for (short nn = 0; nn < TN; nn++) {
      nax_frag_store_int32(c_frags[mm * TN + nn], D, int(N), sc, short(mm * 16),
                           short(nn * 16), M, N, m_base, n_base);
    }
  }
}

kernel void int8_matmul_int32(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device int32_t *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  w8a8_gemm_int32_impl<128, 128, 512, 32, 4, 4>(
      A, B, C, M, N, K, swizzle_log, tiles_m, tiles_n, tgid, sgid, lid);
}

kernel void int8_matmul_int32_small(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device int32_t *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  w8a8_gemm_int32_impl<32, 128, 512, 32, 1, 4>(
      A, B, C, M, N, K, swizzle_log, tiles_m, tiles_n, tgid, sgid, lid);
}
)MTL";

// ── Host: pipeline cache ─────────────────────────────────────────
static id<MTLComputePipelineState> g_pso_large = nil;
static id<MTLComputePipelineState> g_pso_small = nil;
static std::string g_compile_error;

static id<MTLLibrary> build_library() {
  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion4_1;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                         options:opts error:&err];
  if (!lib) {
    g_compile_error = err ? std::string(err.localizedDescription.UTF8String) : "unknown";
    TORCH_CHECK(false, "int8 Metal library compile failed: ", g_compile_error);
  }
  return lib;
}

static id<MTLComputePipelineState> pso_for(bool small) {
  if (small && g_pso_small) return g_pso_small;
  if (!small && g_pso_large) return g_pso_large;
  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  id<MTLLibrary> lib = build_library();
  NSString* name = small ? @"int8_matmul_int32_small" : @"int8_matmul_int32";
  id<MTLFunction> fn = [lib newFunctionWithName:name];
  TORCH_CHECK(fn, "kernel function not found: ", name.UTF8String);
  NSError* err = nil;
  id<MTLComputePipelineState> pso =
      [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "pipeline state creation failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  if (small) g_pso_small = pso; else g_pso_large = pso;
  return pso;
}

// C[M,N] int32 = A[M,K] int8 @ B[N,K]^T int8.  B is the weight in its
// natural [N,K] storage (NT) — no transpose needed.
torch::Tensor i8_matmul2d_nt(torch::Tensor a_i8, torch::Tensor b_i8) {
  TORCH_CHECK(a_i8.is_mps() && b_i8.is_mps(), "inputs must be on mps");
  TORCH_CHECK(a_i8.scalar_type() == torch::kChar && b_i8.scalar_type() == torch::kChar,
              "inputs must be int8");
  TORCH_CHECK(a_i8.dim() == 2 && b_i8.dim() == 2, "inputs must be 2D");
  TORCH_CHECK(a_i8.size(1) == b_i8.size(1), "K mismatch: A[M,K], B[N,K]");
  TORCH_CHECK(a_i8.is_contiguous() && b_i8.is_contiguous(), "inputs must be contiguous");

  const int M = (int)a_i8.size(0);
  const int K = (int)a_i8.size(1);
  const int N = (int)b_i8.size(0);
  auto C = torch::empty({(long)M, (long)N}, a_i8.options().dtype(torch::kInt32));

  // Tile selection mirrors Cider: small tile for M<=64.
  const bool small = (M <= 64);
  const uint BM = small ? 32u : 128u;
  const uint BN = 128u;
  const uint THREADS = small ? 128u : 512u;

  uint tiles_m = (M + BM - 1) / BM;
  uint tiles_n = (N + BN - 1) / BN;
  uint swizzle_log;
  if (tiles_m <= 3) swizzle_log = 0;
  else if (tiles_m <= 6) swizzle_log = 1;
  else swizzle_log = 2;
  uint tile = 1u << swizzle_log;
  uint grid_x = tiles_n * tile;
  uint grid_y = (tiles_m + tile - 1) / tile;

  id<MTLComputePipelineState> pso = pso_for(small);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a_i8.storage().data());
  id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, b_i8.storage().data());
  id<MTLBuffer> cBuf = __builtin_bit_cast(id<MTLBuffer>, C.storage().data());
  const NSUInteger aOff = a_i8.storage_offset() * a_i8.element_size();
  const NSUInteger bOff = b_i8.storage_offset() * b_i8.element_size();
  const NSUInteger cOff = C.storage_offset() * C.element_size();
  uint Mu = (uint)M, Nu = (uint)N, Ku = (uint)K;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:aBuf offset:aOff atIndex:0];
      [enc setBuffer:bBuf offset:bOff atIndex:1];
      [enc setBuffer:cBuf offset:cOff atIndex:2];
      [enc setBytes:&Mu length:sizeof(uint) atIndex:3];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:4];
      [enc setBytes:&Ku length:sizeof(uint) atIndex:5];
      [enc setBytes:&swizzle_log length:sizeof(uint) atIndex:6];
      [enc setBytes:&tiles_m length:sizeof(uint) atIndex:7];
      [enc setBytes:&tiles_n length:sizeof(uint) atIndex:8];
      [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
          threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    }
  });
  // Do NOT COMMIT_AND_WAIT here: the kernel is encoded on torch's MPS stream, so
  // ordering with subsequent torch ops (the rescale that reads C) is preserved on
  // the same command buffer, and torch commits at its normal sync points. Forcing
  // a full GPU sync per call would serialize the whole pipeline (kills async
  // overlap) — that was the cause of the in-model slowdown.
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("i8_matmul2d_nt", &i8_matmul2d_nt,
        "Bit-exact NT int8 matmul: C[M,N] int32 = A[M,K] @ B[N,K]^T on MPS");
  m.def("warmup", []() { pso_for(false); pso_for(true); return true; },
        "compile + build pipelines");
}
