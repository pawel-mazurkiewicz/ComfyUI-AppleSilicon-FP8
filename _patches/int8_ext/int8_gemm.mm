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
// Two epilogues share one GEMM body (w8a8_gemm_compute):
//   • int8_matmul_int32      — stores raw INT32 C (bit-exact vs _int_mm).
//   • int8_matmul_fused      — fuses the W8A8 rescale + optional bias into
//     the store: D[m,n] = bf16(float(C[m,n]) * row_scale[m]) + bf16(bias[n]),
//     so the int32 product never round-trips through global memory (the
//     "fused dequant" variant; cuts the per-call epilogue traffic).
//
// The Metal kernel body (helpers + GEMM compute + entry points) is ported
// from Cider (https://github.com/Mininglamp-AI/cider,
// cider/kernels/w8a8_matmul.metal — including its w8a8_matmul_fused_dequant
// epilogue), MIT License, Copyright (c) 2026 Mininglamp contributors. Only
// the host dispatch is ours (torch MPS ObjC++ shim mirroring
// _patches/fp8_ext/fp8_matmul2d.mm).
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

// ── matmul2d op type, at program scope ──────────────────────────
// Deliberately not function-local. `decltype()` on a local cooperative tensor
// yields a `thread`-qualified type, and macOS 26A5388g tightened
// get_destination_cooperative_tensor with an __is_cooperative_tensor_type_v
// constraint that address-space-qualified types fail -- so the operand types
// have to come from the op's own unqualified member aliases instead (issue
// #13). Those aliases can't be spelled from a function-local descriptor (an
// alias template may not take a local as a template argument), and Metal
// requires a program-scope variable to live in the constant address space.
constant constexpr auto kMMDesc = mpp::tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, true, true,
    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
using gemm_t = mpp::tensor_ops::matmul2d<kMMDesc, metal::execution_simdgroup>;
using ct_a_t = gemm_t::cooperative_tensor_left_input_t<int8_t, int8_t, int32_t>;
using ct_b_t = gemm_t::cooperative_tensor_right_input_t<int8_t, int8_t, int32_t>;

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

// ── Fragment store: fused dequant + optional bias ───────────────
// D[m,n] = OutT(float(C[m,n]) * row_scale[m]) + OutT(bias[n]).
// Matches the eager epilogue exactly: int32→fp32 convert, fp32 rescale by
// the per-row factor (weight_scale * x_row_scale), round to OutT, then a
// SEPARATE OutT-precision bias add (bf16 + bf16) — so it is bit-identical
// to chunk.float()*scale -> .to(bf16) -> + bias.to(bf16).
template <typename OutT>
inline void nax_frag_store_dequant(const thread int32_t *src, device OutT *dst,
                                   int ld, short2 sc, short off_m, short off_n,
                                   uint M, uint N, uint m_base, uint n_base,
                                   const device float *row_scale,
                                   const device OutT *bias, bool has_bias) {
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      uint ni = n_base + sc.x + off_n + j;
      if (mi < M && ni < N) {
        float acc = float(src[i * kElemCols + j]) * row_scale[mi];
        OutT r = OutT(acc);
        if (has_bias) {
          r = r + bias[ni];
        }
        dst[(sc.y + off_m + i * kElemRowsJump) * ld + (sc.x + off_n + j)] = r;
      }
    }
  }
}

// ── Fragment store: fused dequant + optional bias + activation ──
// P0 verdict (M5 Max / macOS 27 / Metal 4.1): precise::exp/precise::tanh compile;
// `erf`/`metal::erf` do NOT -> act=3 (gelu-erf) is removed. Supported act enum:
//   0=none, 1=silu (x*sigmoid(x)), 2=gelu-tanh (F.gelu approximate='tanh').
// Activation is applied AFTER rescale+bias (FFN convention), in fp32 on the
// bf16-rounded post-bias value, then rounded back to OutT.
template <typename OutT>
inline void nax_frag_store_dequant_act(const thread int32_t *src, device OutT *dst,
                                       int ld, short2 sc, short off_m, short off_n,
                                       uint M, uint N, uint m_base, uint n_base,
                                       const device float *row_scale,
                                       const device OutT *bias, bool has_bias, uint act) {
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      uint ni = n_base + sc.x + off_n + j;
      if (mi < M && ni < N) {
        float acc = float(src[i * kElemCols + j]) * row_scale[mi];
        OutT r = OutT(acc);
        if (has_bias) { r = r + bias[ni]; }
        float y = float(r);                 // promote the bf16 linear output to fp32
        if (act == 1u) {                    // SiLU: x * sigmoid(x)
          y = y / (1.0f + precise::exp(-y));
        } else if (act == 2u) {             // GELU tanh-approx (F.gelu approximate='tanh')
          // tanh(z)=2*sigmoid(2z)-1 => 0.5*x*(1+tanh(z)) == x*sigmoid(2z)
          //  = x / (1 + exp(-2z)).  Metal precise::tanh is low-accuracy (~160 ulp
          // off torch), but precise::exp matches torch (silu is bit-exact to 1 ulp),
          // so route GELU-tanh through exp.  z = sqrt(2/pi)*(x + 0.044715 x^3).
          float c2 = 1.5957691216057308f;   // 2*sqrt(2/pi)
          y = y / (1.0f + precise::exp(-c2 * (y + 0.044715f * y * y * y)));
        }
        dst[(sc.y + off_m + i * kElemRowsJump) * ld + (sc.x + off_n + j)] = OutT(y);
      }
    }
  }
}

// ── Fragment store: fused SwiGLU/GEGLU gate ─────────────────────
// H[m,n] = act(gate[m,n]*rs_g[m] + bias_g[n]) * (up[m,n]*rs_u[m] + bias_u[n]),
// gate/up being two int32 register accumulators for the SAME (m,n) tile from two
// GEMMs (B=Wg, B=Wu). act: 1=silu (SwiGLU), 2=gelu-tanh (GEGLU). GELU via the exp
// identity (precise::tanh is low-accuracy; see nax_frag_store_dequant_act).
template <typename OutT>
inline void nax_frag_store_swiglu(const thread int32_t *gate, const thread int32_t *up,
                                  device OutT *dst, int ld, short2 sc, short off_m, short off_n,
                                  uint M, uint N, uint m_base, uint n_base,
                                  const device float *rs_g, const device float *rs_u,
                                  const device OutT *bias_g, const device OutT *bias_u,
                                  bool hb_g, bool hb_u, uint act) {
  for (short i = 0; i < 2; i++) {
    for (short j = 0; j < kElemCols; j++) {
      uint mi = m_base + sc.y + off_m + i * kElemRowsJump;
      uint ni = n_base + sc.x + off_n + j;
      if (mi < M && ni < N) {
        short e = i * kElemCols + j;
        OutT g = OutT(float(gate[e]) * rs_g[mi]); if (hb_g) g = g + bias_g[ni];
        OutT u = OutT(float(up[e])   * rs_u[mi]); if (hb_u) u = u + bias_u[ni];
        float gf = float(g);
        if (act == 1u) {                              // SwiGLU
          gf = gf / (1.0f + precise::exp(-gf));
        } else if (act == 2u) {                       // GEGLU-tanh (via exp identity)
          float c2 = 1.5957691216057308f;             // 2*sqrt(2/pi)
          gf = gf / (1.0f + precise::exp(-c2 * (gf + 0.044715f * gf * gf * gf)));
        }
        dst[(sc.y + off_m + i * kElemRowsJump) * ld + (sc.x + off_n + j)] = OutT(gf * float(u));
      }
    }
  }
}

// ── Raw INT32 GEMM compute (B is [N, K], transpose_b=true) ──────
// Fills c_frags (int32 register accumulators) for this simdgroup's tile
// of C[M,N] = A[M,K] × B[N,K]^T, and reports the tile origin (m_base,
// n_base) + per-thread NAXFrag coord (sc) for the caller's store epilogue.
// Returns false when this threadgroup maps outside the problem (skip store).
template <int BM, int BN, int BK, int SK, int WM, int WN>
bool w8a8_gemm_compute(
    const device int8_t *A, const device int8_t *B,
    thread int32_t (&c_frags)[((BM / WM) / 16) * ((BN / WN) / 16)][kElemsPerFrag],
    thread short2 &sc_out, thread uint &m_base_out, thread uint &n_base_out,
    uint M, uint N, uint K, uint swizzle_log, uint tiles_m, uint tiles_n,
    uint2 tgid, uint sgid, uint lid) {
  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  uint tid_y = (tgid.y << swizzle_log) + (tgid.x & ((1u << swizzle_log) - 1u));
  uint tid_x = tgid.x >> swizzle_log;
  if (tid_x >= tiles_n || tid_y >= tiles_m) {
    return false;
  }

  short2 sc = nax_get_coord(ushort(lid));
  uint sg_row = sgid / WN;
  uint sg_col = sgid % WN;
  uint m_base = tid_y * BM + sg_row * SM;
  uint n_base = tid_x * BN + sg_col * SN;

  const device int8_t *sg_A = A + m_base * K;
  const device int8_t *sg_B = B + n_base * K;

  gemm_t gemm_op;

  auto ct_a =
      gemm_op.get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      gemm_op.get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_c =
      gemm_op.get_destination_cooperative_tensor<ct_a_t, ct_b_t, int32_t>();

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

  sc_out = sc;
  m_base_out = m_base;
  n_base_out = n_base;
  return true;
}

// ── INT32 entry points (raw store) ──────────────────────────────
kernel void int8_matmul_int32(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device int32_t *C [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4>(
          A, B, c_frags, sc, m_base, n_base, M, N, K, swizzle_log, tiles_m,
          tiles_n, tgid, sgid, lid))
    return;
  device int32_t *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_int32(c_frags[mm * TN + nn], D, int(N), sc, short(mm * 16),
                           short(nn * 16), M, N, m_base, n_base);
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
  constexpr short TM = ((32 / 1) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<32, 128, 512, 32, 1, 4>(
          A, B, c_frags, sc, m_base, n_base, M, N, K, swizzle_log, tiles_m,
          tiles_n, tgid, sgid, lid))
    return;
  device int32_t *D = C + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_int32(c_frags[mm * TN + nn], D, int(N), sc, short(mm * 16),
                           short(nn * 16), M, N, m_base, n_base);
}

// ── Fused (dequant + bias) entry points, bf16 output ────────────
kernel void int8_matmul_fused(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device bfloat *D [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    const device float *row_scale [[buffer(9)]],
    const device bfloat *bias [[buffer(10)]],
    constant uint &has_bias [[buffer(11)]],
    constant uint &act [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4>(
          A, B, c_frags, sc, m_base, n_base, M, N, K, swizzle_log, tiles_m,
          tiles_n, tgid, sgid, lid))
    return;
  device bfloat *Dp = D + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      if (act == 0u)
        nax_frag_store_dequant<bfloat>(c_frags[mm * TN + nn], Dp, int(N), sc,
                                       short(mm * 16), short(nn * 16), M, N,
                                       m_base, n_base, row_scale, bias,
                                       has_bias != 0u);
      else
        nax_frag_store_dequant_act<bfloat>(c_frags[mm * TN + nn], Dp, int(N), sc,
                                           short(mm * 16), short(nn * 16), M, N,
                                           m_base, n_base, row_scale, bias,
                                           has_bias != 0u, act);
}

kernel void int8_matmul_fused_small(
    const device int8_t *A [[buffer(0)]], const device int8_t *B [[buffer(1)]],
    device bfloat *D [[buffer(2)]], constant uint &M [[buffer(3)]],
    constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
    constant uint &swizzle_log [[buffer(6)]],
    constant uint &tiles_m [[buffer(7)]], constant uint &tiles_n [[buffer(8)]],
    const device float *row_scale [[buffer(9)]],
    const device bfloat *bias [[buffer(10)]],
    constant uint &has_bias [[buffer(11)]],
    constant uint &act [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((32 / 1) / 16), TN = ((128 / 4) / 16);
  int32_t c_frags[TM * TN][kElemsPerFrag];
  short2 sc;
  uint m_base, n_base;
  if (!w8a8_gemm_compute<32, 128, 512, 32, 1, 4>(
          A, B, c_frags, sc, m_base, n_base, M, N, K, swizzle_log, tiles_m,
          tiles_n, tgid, sgid, lid))
    return;
  device bfloat *Dp = D + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      if (act == 0u)
        nax_frag_store_dequant<bfloat>(c_frags[mm * TN + nn], Dp, int(N), sc,
                                       short(mm * 16), short(nn * 16), M, N,
                                       m_base, n_base, row_scale, bias,
                                       has_bias != 0u);
      else
        nax_frag_store_dequant_act<bfloat>(c_frags[mm * TN + nn], Dp, int(N), sc,
                                           short(mm * 16), short(nn * 16), M, N,
                                           m_base, n_base, row_scale, bias,
                                           has_bias != 0u, act);
}

// ── Fused SwiGLU/GEGLU gate entry points, bf16 output ───────────
// Runs the templated GEMM body twice for the SAME output tile (B=Bg, then
// B=Bu) into two int32 register accumulators, then one combined gated store:
// D = act(A@Bg^T*rs_g + bias_g) * (A@Bu^T*rs_u + bias_u).
kernel void int8_matmul_swiglu(
    const device int8_t *A [[buffer(0)]], const device int8_t *Bg [[buffer(1)]],
    const device int8_t *Bu [[buffer(2)]], device bfloat *D [[buffer(3)]],
    constant uint &M [[buffer(4)]], constant uint &N [[buffer(5)]], constant uint &K [[buffer(6)]],
    constant uint &swizzle_log [[buffer(7)]], constant uint &tiles_m [[buffer(8)]],
    constant uint &tiles_n [[buffer(9)]], const device float *rs_g [[buffer(10)]],
    const device float *rs_u [[buffer(11)]], const device bfloat *bias_g [[buffer(12)]],
    const device bfloat *bias_u [[buffer(13)]], constant uint &hb_g [[buffer(14)]],
    constant uint &hb_u [[buffer(15)]], constant uint &act [[buffer(16)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((128 / 4) / 16), TN = ((128 / 4) / 16);
  int32_t cg[TM * TN][kElemsPerFrag];
  short2 sc; uint m_base, n_base;
  if (!w8a8_gemm_compute<128, 128, 512, 32, 4, 4>(A, Bg, cg, sc, m_base, n_base, M, N, K,
        swizzle_log, tiles_m, tiles_n, tgid, sgid, lid)) return;
  int32_t cu[TM * TN][kElemsPerFrag];
  short2 sc2; uint mb2, nb2;
  bool ok2 = w8a8_gemm_compute<128, 128, 512, 32, 4, 4>(A, Bu, cu, sc2, mb2, nb2, M, N, K,
        swizzle_log, tiles_m, tiles_n, tgid, sgid, lid);
  if (!ok2 || sc2.x != sc.x || sc2.y != sc.y || mb2 != m_base || nb2 != n_base) return;
  device bfloat *Dp = D + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_swiglu<bfloat>(cg[mm * TN + nn], cu[mm * TN + nn], Dp, int(N), sc,
        short(mm * 16), short(nn * 16), M, N, m_base, n_base, rs_g, rs_u, bias_g, bias_u,
        hb_g != 0u, hb_u != 0u, act);
}

kernel void int8_matmul_swiglu_small(
    const device int8_t *A [[buffer(0)]], const device int8_t *Bg [[buffer(1)]],
    const device int8_t *Bu [[buffer(2)]], device bfloat *D [[buffer(3)]],
    constant uint &M [[buffer(4)]], constant uint &N [[buffer(5)]], constant uint &K [[buffer(6)]],
    constant uint &swizzle_log [[buffer(7)]], constant uint &tiles_m [[buffer(8)]],
    constant uint &tiles_n [[buffer(9)]], const device float *rs_g [[buffer(10)]],
    const device float *rs_u [[buffer(11)]], const device bfloat *bias_g [[buffer(12)]],
    const device bfloat *bias_u [[buffer(13)]], constant uint &hb_g [[buffer(14)]],
    constant uint &hb_u [[buffer(15)]], constant uint &act [[buffer(16)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]]) {
  constexpr short TM = ((32 / 1) / 16), TN = ((128 / 4) / 16);
  int32_t cg[TM * TN][kElemsPerFrag];
  short2 sc; uint m_base, n_base;
  if (!w8a8_gemm_compute<32, 128, 512, 32, 1, 4>(A, Bg, cg, sc, m_base, n_base, M, N, K,
        swizzle_log, tiles_m, tiles_n, tgid, sgid, lid)) return;
  int32_t cu[TM * TN][kElemsPerFrag];
  short2 sc2; uint mb2, nb2;
  bool ok2 = w8a8_gemm_compute<32, 128, 512, 32, 1, 4>(A, Bu, cu, sc2, mb2, nb2, M, N, K,
        swizzle_log, tiles_m, tiles_n, tgid, sgid, lid);
  if (!ok2 || sc2.x != sc.x || sc2.y != sc.y || mb2 != m_base || nb2 != n_base) return;
  device bfloat *Dp = D + m_base * N + n_base;
  for (short mm = 0; mm < TM; mm++)
    for (short nn = 0; nn < TN; nn++)
      nax_frag_store_swiglu<bfloat>(cg[mm * TN + nn], cu[mm * TN + nn], Dp, int(N), sc,
        short(mm * 16), short(nn * 16), M, N, m_base, n_base, rs_g, rs_u, bias_g, bias_u,
        hb_g != 0u, hb_u != 0u, act);
}
)MTL";

// ── Host: pipeline cache ─────────────────────────────────────────
static id<MTLComputePipelineState> g_pso_large = nil;
static id<MTLComputePipelineState> g_pso_small = nil;
static id<MTLComputePipelineState> g_pso_fused = nil;
static id<MTLComputePipelineState> g_pso_fused_small = nil;
static id<MTLComputePipelineState> g_pso_swiglu = nil;
static id<MTLComputePipelineState> g_pso_swiglu_small = nil;
static id<MTLLibrary> g_lib = nil;
static std::string g_compile_error;
// Latch failure too, not just success: the library is compiled on first dispatch,
// so a toolchain that rejects it would otherwise recompile on every call (issue #13).
static bool g_compile_failed = false;

static id<MTLLibrary> build_library() {
  if (g_lib) return g_lib;
  TORCH_CHECK(!g_compile_failed,
              "int8 Metal library compile failed (cached): ", g_compile_error);
  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  MTLCompileOptions* opts = [MTLCompileOptions new];
  opts.languageVersion = MTLLanguageVersion4_1;
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                         options:opts error:&err];
  if (!lib) {
    g_compile_error = err ? std::string(err.localizedDescription.UTF8String) : "unknown";
    g_compile_failed = true;
    TORCH_CHECK(false, "int8 Metal library compile failed: ", g_compile_error);
  }
  g_lib = lib;
  return lib;
}

static id<MTLComputePipelineState> pso_for_name(NSString* name) {
  id<MTLDevice> dev = MPSDevice::getInstance()->device();
  id<MTLLibrary> lib = build_library();
  id<MTLFunction> fn = [lib newFunctionWithName:name];
  TORCH_CHECK(fn, "kernel function not found: ", name.UTF8String);
  NSError* err = nil;
  id<MTLComputePipelineState> pso =
      [dev newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso, "pipeline state creation failed: ",
              err ? err.localizedDescription.UTF8String : "unknown");
  return pso;
}

static id<MTLComputePipelineState> pso_for(bool small) {
  if (small && g_pso_small) return g_pso_small;
  if (!small && g_pso_large) return g_pso_large;
  id<MTLComputePipelineState> pso =
      pso_for_name(small ? @"int8_matmul_int32_small" : @"int8_matmul_int32");
  if (small) g_pso_small = pso; else g_pso_large = pso;
  return pso;
}

static id<MTLComputePipelineState> pso_for_fused(bool small) {
  if (small && g_pso_fused_small) return g_pso_fused_small;
  if (!small && g_pso_fused) return g_pso_fused;
  id<MTLComputePipelineState> pso =
      pso_for_name(small ? @"int8_matmul_fused_small" : @"int8_matmul_fused");
  if (small) g_pso_fused_small = pso; else g_pso_fused = pso;
  return pso;
}

static id<MTLComputePipelineState> pso_for_swiglu(bool small) {
  if (small && g_pso_swiglu_small) return g_pso_swiglu_small;
  if (!small && g_pso_swiglu) return g_pso_swiglu;
  id<MTLComputePipelineState> pso =
      pso_for_name(small ? @"int8_matmul_swiglu_small" : @"int8_matmul_swiglu");
  if (small) g_pso_swiglu_small = pso; else g_pso_swiglu = pso;
  return pso;
}

// Shared grid/swizzle geometry for both epilogues.
struct Geom {
  bool small;
  uint THREADS, swizzle_log, grid_x, grid_y, tiles_m, tiles_n;
};

static Geom geom_for(int M, int N) {
  Geom g;
  g.small = (M <= 64);
  const uint BM = g.small ? 32u : 128u;
  const uint BN = 128u;
  g.THREADS = g.small ? 128u : 512u;
  g.tiles_m = (M + BM - 1) / BM;
  g.tiles_n = (N + BN - 1) / BN;
  if (g.tiles_m <= 3) g.swizzle_log = 0;
  else if (g.tiles_m <= 6) g.swizzle_log = 1;
  else g.swizzle_log = 2;
  uint tile = 1u << g.swizzle_log;
  g.grid_x = g.tiles_n * tile;
  g.grid_y = (g.tiles_m + tile - 1) / tile;
  return g;
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

  Geom g = geom_for(M, N);
  id<MTLComputePipelineState> pso = pso_for(g.small);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a_i8.storage().data());
  id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, b_i8.storage().data());
  id<MTLBuffer> cBuf = __builtin_bit_cast(id<MTLBuffer>, C.storage().data());
  const NSUInteger aOff = a_i8.storage_offset() * a_i8.element_size();
  const NSUInteger bOff = b_i8.storage_offset() * b_i8.element_size();
  const NSUInteger cOff = C.storage_offset() * C.element_size();
  uint Mu = (uint)M, Nu = (uint)N, Ku = (uint)K;
  uint swizzle_log = g.swizzle_log, tiles_m = g.tiles_m, tiles_n = g.tiles_n;

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
      [enc dispatchThreadgroups:MTLSizeMake(g.grid_x, g.grid_y, 1)
          threadsPerThreadgroup:MTLSizeMake(g.THREADS, 1, 1)];
    }
  });
  // Do NOT COMMIT_AND_WAIT here: the kernel is encoded on torch's MPS stream, so
  // ordering with subsequent torch ops (the rescale that reads C) is preserved on
  // the same command buffer, and torch commits at its normal sync points. Forcing
  // a full GPU sync per call would serialize the whole pipeline (kills async
  // overlap) — that was the cause of the in-model slowdown.
  return C;
}

// Fused: D[M,N] bf16 = (A[M,K] int8 @ B[N,K]^T int8) rescaled per-row by
// row_scale[M] (= weight_scale * activation_row_scale, fp32) + optional
// bias[N] (bf16). The int32 product is consumed in registers and never
// written to global memory — this is the per-call epilogue we skip.
torch::Tensor i8_matmul2d_nt_fused(torch::Tensor a_i8, torch::Tensor b_i8,
                                   torch::Tensor row_scale,
                                   c10::optional<torch::Tensor> bias_opt,
                                   int64_t act) {
  TORCH_CHECK(a_i8.is_mps() && b_i8.is_mps() && row_scale.is_mps(),
              "inputs must be on mps");
  TORCH_CHECK(a_i8.scalar_type() == torch::kChar && b_i8.scalar_type() == torch::kChar,
              "a/b must be int8");
  TORCH_CHECK(a_i8.dim() == 2 && b_i8.dim() == 2, "a/b must be 2D");
  TORCH_CHECK(a_i8.size(1) == b_i8.size(1), "K mismatch: A[M,K], B[N,K]");
  TORCH_CHECK(a_i8.is_contiguous() && b_i8.is_contiguous(), "a/b must be contiguous");
  TORCH_CHECK(row_scale.scalar_type() == torch::kFloat && row_scale.is_contiguous(),
              "row_scale must be contiguous float32");
  TORCH_CHECK(row_scale.numel() == a_i8.size(0), "row_scale must have M elements");
  // Defense-in-depth: the store epilogue only branches on act 1 (silu) / 2 (gelu-tanh);
  // act 0 means "none". Any other value would silently emit un-activated output, so reject
  // it here (act=3 gelu-erf was dropped — Metal `erf` does not compile under 4.1).
  TORCH_CHECK(act >= 0 && act <= 2, "fused: act must be 0(none)/1(silu)/2(gelu-tanh)");

  const int M = (int)a_i8.size(0);
  const int K = (int)a_i8.size(1);
  const int N = (int)b_i8.size(0);
  auto D = torch::empty({(long)M, (long)N}, a_i8.options().dtype(torch::kBFloat16));

  const bool has_bias = bias_opt.has_value();
  torch::Tensor bias = has_bias
      ? bias_opt.value().to(torch::kBFloat16).contiguous()
      : torch::zeros({1}, a_i8.options().dtype(torch::kBFloat16));
  if (has_bias) {
    TORCH_CHECK(bias.is_mps() && bias.numel() == N, "bias must be N MPS elements");
  }

  Geom g = geom_for(M, N);
  id<MTLComputePipelineState> pso = pso_for_fused(g.small);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a_i8.storage().data());
  id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, b_i8.storage().data());
  id<MTLBuffer> dBuf = __builtin_bit_cast(id<MTLBuffer>, D.storage().data());
  id<MTLBuffer> sBuf = __builtin_bit_cast(id<MTLBuffer>, row_scale.storage().data());
  id<MTLBuffer> biasBuf = __builtin_bit_cast(id<MTLBuffer>, bias.storage().data());
  const NSUInteger aOff = a_i8.storage_offset() * a_i8.element_size();
  const NSUInteger bOff = b_i8.storage_offset() * b_i8.element_size();
  const NSUInteger dOff = D.storage_offset() * D.element_size();
  const NSUInteger sOff = row_scale.storage_offset() * row_scale.element_size();
  const NSUInteger biasOff = bias.storage_offset() * bias.element_size();
  uint Mu = (uint)M, Nu = (uint)N, Ku = (uint)K;
  uint swizzle_log = g.swizzle_log, tiles_m = g.tiles_m, tiles_n = g.tiles_n;
  uint has_bias_u = has_bias ? 1u : 0u;
  uint act_u = (uint)act;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:aBuf offset:aOff atIndex:0];
      [enc setBuffer:bBuf offset:bOff atIndex:1];
      [enc setBuffer:dBuf offset:dOff atIndex:2];
      [enc setBytes:&Mu length:sizeof(uint) atIndex:3];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:4];
      [enc setBytes:&Ku length:sizeof(uint) atIndex:5];
      [enc setBytes:&swizzle_log length:sizeof(uint) atIndex:6];
      [enc setBytes:&tiles_m length:sizeof(uint) atIndex:7];
      [enc setBytes:&tiles_n length:sizeof(uint) atIndex:8];
      [enc setBuffer:sBuf offset:sOff atIndex:9];
      [enc setBuffer:biasBuf offset:biasOff atIndex:10];
      [enc setBytes:&has_bias_u length:sizeof(uint) atIndex:11];
      [enc setBytes:&act_u length:sizeof(uint) atIndex:12];
      [enc dispatchThreadgroups:MTLSizeMake(g.grid_x, g.grid_y, 1)
          threadsPerThreadgroup:MTLSizeMake(g.THREADS, 1, 1)];
    }
  });
  return D;
}

// Fused gated: D[M,N] bf16 = act(A@Bg^T*rs_g[M] + bias_g[N]) * (A@Bu^T*rs_u[M] +
// bias_u[N]). One launch, two register accumulators, two bf16 intermediates that
// never leave registers. act: 1=silu (SwiGLU), 2=gelu-tanh (GEGLU).
torch::Tensor i8_matmul2d_nt_swiglu(torch::Tensor a_i8, torch::Tensor bg_i8,
                                    torch::Tensor bu_i8, torch::Tensor row_scale_gate,
                                    torch::Tensor row_scale_up,
                                    c10::optional<torch::Tensor> bias_gate,
                                    c10::optional<torch::Tensor> bias_up,
                                    int64_t act) {
  TORCH_CHECK(a_i8.is_mps() && bg_i8.is_mps() && bu_i8.is_mps(), "swiglu: inputs must be MPS");
  TORCH_CHECK(a_i8.scalar_type() == torch::kChar && bg_i8.scalar_type() == torch::kChar
              && bu_i8.scalar_type() == torch::kChar, "swiglu: A/Bg/Bu must be int8");
  TORCH_CHECK(a_i8.dim() == 2 && bg_i8.dim() == 2 && bu_i8.dim() == 2, "swiglu: A/Bg/Bu must be 2D");
  TORCH_CHECK(bg_i8.sizes() == bu_i8.sizes(), "swiglu: Bg and Bu must have identical [N,K]");
  TORCH_CHECK(a_i8.size(1) == bg_i8.size(1), "swiglu: K mismatch A vs Bg/Bu");
  TORCH_CHECK(a_i8.is_contiguous() && bg_i8.is_contiguous() && bu_i8.is_contiguous(),
              "swiglu: A/Bg/Bu must be contiguous");
  const int64_t Mll = a_i8.size(0), Nll = bg_i8.size(0);
  TORCH_CHECK(row_scale_gate.is_contiguous() && row_scale_up.is_contiguous()
              && row_scale_gate.scalar_type() == torch::kFloat
              && row_scale_up.scalar_type() == torch::kFloat
              && row_scale_gate.numel() == Mll && row_scale_up.numel() == Mll,
              "swiglu: row scales must be contiguous fp32 of length M");

  const int M = (int)Mll;
  const int K = (int)a_i8.size(1);
  const int N = (int)Nll;
  auto D = torch::empty({(long)M, (long)N}, a_i8.options().dtype(torch::kBFloat16));

  const bool hb_g = bias_gate.has_value();
  const bool hb_u = bias_up.has_value();
  torch::Tensor bg = hb_g ? bias_gate.value().to(torch::kBFloat16).contiguous()
                          : torch::zeros({1}, a_i8.options().dtype(torch::kBFloat16));
  torch::Tensor bu = hb_u ? bias_up.value().to(torch::kBFloat16).contiguous()
                          : torch::zeros({1}, a_i8.options().dtype(torch::kBFloat16));
  if (hb_g) TORCH_CHECK(bg.is_mps() && bg.numel() == N, "swiglu: bias_gate must be N MPS elements");
  if (hb_u) TORCH_CHECK(bu.is_mps() && bu.numel() == N, "swiglu: bias_up must be N MPS elements");
  // Defense-in-depth: the gated store only branches on act 1 (SwiGLU) / 2 (GEGLU-tanh);
  // act 0 / anything else would gate by the raw linear value (un-activated), so reject it.
  TORCH_CHECK(act == 1 || act == 2, "swiglu: act must be 1(silu) or 2(gelu-tanh)");

  Geom g = geom_for(M, N);
  id<MTLComputePipelineState> pso = pso_for_swiglu(g.small);
  MPSStream* stream = getCurrentMPSStream();
  id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, a_i8.storage().data());
  id<MTLBuffer> bgBuf = __builtin_bit_cast(id<MTLBuffer>, bg_i8.storage().data());
  id<MTLBuffer> buBuf = __builtin_bit_cast(id<MTLBuffer>, bu_i8.storage().data());
  id<MTLBuffer> dBuf = __builtin_bit_cast(id<MTLBuffer>, D.storage().data());
  id<MTLBuffer> sgBuf = __builtin_bit_cast(id<MTLBuffer>, row_scale_gate.storage().data());
  id<MTLBuffer> suBuf = __builtin_bit_cast(id<MTLBuffer>, row_scale_up.storage().data());
  id<MTLBuffer> bgBiasBuf = __builtin_bit_cast(id<MTLBuffer>, bg.storage().data());
  id<MTLBuffer> buBiasBuf = __builtin_bit_cast(id<MTLBuffer>, bu.storage().data());
  const NSUInteger aOff = a_i8.storage_offset() * a_i8.element_size();
  const NSUInteger bgOff = bg_i8.storage_offset() * bg_i8.element_size();
  const NSUInteger buOff = bu_i8.storage_offset() * bu_i8.element_size();
  const NSUInteger dOff = D.storage_offset() * D.element_size();
  const NSUInteger sgOff = row_scale_gate.storage_offset() * row_scale_gate.element_size();
  const NSUInteger suOff = row_scale_up.storage_offset() * row_scale_up.element_size();
  const NSUInteger bgBiasOff = bg.storage_offset() * bg.element_size();
  const NSUInteger buBiasOff = bu.storage_offset() * bu.element_size();
  uint Mu = (uint)M, Nu = (uint)N, Ku = (uint)K;
  uint swizzle_log = g.swizzle_log, tiles_m = g.tiles_m, tiles_n = g.tiles_n;
  uint hb_g_u = hb_g ? 1u : 0u, hb_u_u = hb_u ? 1u : 0u, act_u = (uint)act;

  dispatch_sync(stream->queue(), ^(){
    @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
      [enc setComputePipelineState:pso];
      [enc setBuffer:aBuf offset:aOff atIndex:0];
      [enc setBuffer:bgBuf offset:bgOff atIndex:1];
      [enc setBuffer:buBuf offset:buOff atIndex:2];
      [enc setBuffer:dBuf offset:dOff atIndex:3];
      [enc setBytes:&Mu length:sizeof(uint) atIndex:4];
      [enc setBytes:&Nu length:sizeof(uint) atIndex:5];
      [enc setBytes:&Ku length:sizeof(uint) atIndex:6];
      [enc setBytes:&swizzle_log length:sizeof(uint) atIndex:7];
      [enc setBytes:&tiles_m length:sizeof(uint) atIndex:8];
      [enc setBytes:&tiles_n length:sizeof(uint) atIndex:9];
      [enc setBuffer:sgBuf offset:sgOff atIndex:10];
      [enc setBuffer:suBuf offset:suOff atIndex:11];
      [enc setBuffer:bgBiasBuf offset:bgBiasOff atIndex:12];
      [enc setBuffer:buBiasBuf offset:buBiasOff atIndex:13];
      [enc setBytes:&hb_g_u length:sizeof(uint) atIndex:14];
      [enc setBytes:&hb_u_u length:sizeof(uint) atIndex:15];
      [enc setBytes:&act_u length:sizeof(uint) atIndex:16];
      [enc dispatchThreadgroups:MTLSizeMake(g.grid_x, g.grid_y, 1)
          threadsPerThreadgroup:MTLSizeMake(g.THREADS, 1, 1)];
    }
  });
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("i8_matmul2d_nt", &i8_matmul2d_nt,
        "Bit-exact NT int8 matmul: C[M,N] int32 = A[M,K] @ B[N,K]^T on MPS");
  m.def("i8_matmul2d_nt_fused", &i8_matmul2d_nt_fused,
        "Fused NT int8 matmul: D[M,N] bf16 = act((A@B^T)*row_scale[M] + bias[N]); "
        "act 0=none,1=silu,2=gelu-tanh",
        pybind11::arg("a_i8"), pybind11::arg("b_i8"), pybind11::arg("row_scale"),
        pybind11::arg("bias") = c10::optional<torch::Tensor>(),
        pybind11::arg("act") = 0);
  m.def("i8_matmul2d_nt_swiglu", &i8_matmul2d_nt_swiglu,
        "Fused gated int8 matmul: D[M,N] bf16 = act(A@Bg^T*rsg+bg) * (A@Bu^T*rsu+bu); "
        "act 1=silu (SwiGLU), 2=gelu-tanh (GEGLU)",
        pybind11::arg("a_i8"), pybind11::arg("bg_i8"), pybind11::arg("bu_i8"),
        pybind11::arg("row_scale_gate"), pybind11::arg("row_scale_up"),
        pybind11::arg("bias_gate") = c10::optional<torch::Tensor>(),
        pybind11::arg("bias_up") = c10::optional<torch::Tensor>(),
        pybind11::arg("act") = 1);
  m.def("warmup", []() {
        pso_for(false); pso_for(true);
        pso_for_fused(false); pso_for_fused(true);
        pso_for_swiglu(false); pso_for_swiglu(true);
        return true;
      },
      "compile + build pipelines");
}
