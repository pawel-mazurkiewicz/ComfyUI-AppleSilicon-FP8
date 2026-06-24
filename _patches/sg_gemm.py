"""Threadgroup-staged simdgroup-matrix bf16 GEMM via torch.mps.compile_shader.

Stages A and B K-tiles into threadgroup memory (as half, since simdgroup_half8x8
is the proven type — simdgroup_bfloat8x8 is not supported on this SDK) and
accumulates with simdgroup_float8x8 into fp32. Proven idiom from
mtlflashattn/_kernel.py. Single-buffered; double-buffering is a later task.
bf16 inputs are reinterpreted as half for staging; fp32 output.

Off-MPS / unsupported SDKs are gated by available()/self_check_ok(); any
failure disables the backend.
"""

import torch

TAG = "[AppleSilicon-FP8/sg_gemm]"

# BM = 8 * NSG (each simdgroup owns 8 output rows). BN, BK multiples of 8.
# NSG=8 → BM=64; BN=64, BK=32 → threadgroup memory = 64*32 + 32*64 = 4096 halfs = 8 KB
_BM, _BN, _BK, _NSG = 64, 64, 32, 8

_GEMM_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr int BM  = @BM@;
constant constexpr int BN  = @BN@;
constant constexpr int BK  = @BK@;
constant constexpr int NSG = @NSG@;

kernel void sgemm(
    device bfloat* A  [[buffer(0)]],   // [M,K] row-major bf16
    device bfloat* B  [[buffer(1)]],   // [K,N] row-major bf16
    device float*  C  [[buffer(2)]],   // [M,N] row-major fp32
    device int*    SH [[buffer(3)]],   // [M,N,K]
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int M = SH[0], N = SH[1], K = SH[2];
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;

    // Threadgroup staging buffers — use half (simdgroup_half8x8 is what the
    // Metal SDK exposes; simdgroup_bfloat8x8 is not available).
    threadgroup half As[BM * BK];   // [BM, BK]
    threadgroup half Bs[BK * BN];   // [BK, BN]

    // Each simdgroup owns 8 consecutive output rows.
    const int sr0 = int(sgid) * 8;   // first output row for this simdgroup

    // Accumulators: BN/8 tiles of 8x8 fp32 per simdgroup.
    simdgroup_float8x8 acc[BN / 8];
    for (int j = 0; j < BN / 8; ++j)
        acc[j] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    const int nthreads = NSG * 32;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // --- Load A tile [BM, BK] ---
        for (int i = int(tid); i < BM * BK; i += nthreads) {
            int r = i / BK, c = i % BK;
            int gr = m0 + r, gc = k0 + c;
            As[i] = half((gr < M && gc < K) ? float(A[ulong(gr) * K + gc]) : 0.0f);
        }
        // --- Load B tile [BK, BN] ---
        for (int i = int(tid); i < BK * BN; i += nthreads) {
            int r = i / BN, c = i % BN;
            int gr = k0 + r, gc = n0 + c;
            Bs[i] = half((gr < K && gc < N) ? float(B[ulong(gr) * N + gc]) : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Compute: step through K-tile in 8-wide chunks ---
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 Amat;
            // A sub-tile: row sr0, col kk; leading dim = BK (cols in As)
            simdgroup_load(Amat, &As[sr0 * BK + kk], ulong(BK));

            for (int j = 0; j < BN / 8; ++j) {
                simdgroup_half8x8 Bmat;
                // B sub-tile: row kk, col j*8; leading dim = BN (cols in Bs)
                simdgroup_load(Bmat, &Bs[kk * BN + j * 8], ulong(BN));
                simdgroup_multiply_accumulate(acc[j], Amat, Bmat, acc[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Store: each simdgroup writes its 8x(BN) block to C ---
    // Use a per-simdgroup threadgroup scratch (sized 8*8) to extract values.
    // We need one tile at a time; use a small static scratch per simdgroup.
    // Since simdgroups execute independently we need separate scratch regions.
    threadgroup float tmp[NSG * 64];   // [NSG, 8, 8]

    for (int j = 0; j < BN / 8; ++j) {
        // Store this 8x8 tile into scratch at offset sgid*64
        simdgroup_store(acc[j], &tmp[int(sgid) * 64], ulong(8));
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Each lane writes up to 2 elements (64 elements / 32 lanes)
        for (int e = int(lane); e < 64; e += 32) {
            int r = e / 8, c = e % 8;
            int gr = m0 + sr0 + r;
            int gc = n0 + j * 8 + c;
            if (gr < M && gc < N)
                C[ulong(gr) * N + gc] = tmp[int(sgid) * 64 + r * 8 + c];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"""

_lib = None
_compiled = None        # None=untried, False=failed, True=ok
_self_check = None      # None=untried, then bool


def _get_lib():
    global _lib, _compiled
    if _compiled is not None:
        return _lib if _compiled else None
    if not hasattr(torch.mps, "compile_shader"):
        _compiled = False
        return None
    try:
        src = (_GEMM_MSL
               .replace("@BM@", str(_BM)).replace("@BN@", str(_BN))
               .replace("@BK@", str(_BK)).replace("@NSG@", str(_NSG)))
        _lib = torch.mps.compile_shader(src)
        _compiled = True
    except Exception as e:
        print(f"{TAG} kernel did not compile; sg_gemm disabled ({e!r}).")
        _lib, _compiled = None, False
    return _lib


def available():
    """True iff the simdgroup GEMM kernel compiles on this OS/SDK/PyTorch build."""
    return _get_lib() is not None


def sg_matmul(a, b):
    """C[M,N] f32 = A[M,K] @ B[K,N]; a,b are bf16 on MPS.

    Inputs are made contiguous so callers need not worry about strides.
    """
    lib = _get_lib()
    if lib is None:
        raise RuntimeError("sg_gemm unavailable")
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"inner dims disagree: {a.shape} @ {b.shape}"
    c = torch.zeros(M, N, device="mps", dtype=torch.float32)
    sh = torch.tensor([M, N, K], dtype=torch.int32, device="mps")
    ntg_x = -(-M // _BM)
    ntg_y = -(-N // _BN)
    # threads = total threads per dimension; group_size = per-threadgroup size.
    # Threadgroups = threads / group_size.
    # X: ntg_x threadgroups cover M tiles; Y: ntg_y threadgroups cover N tiles.
    lib.sgemm(a, b, c, sh,
              threads=(ntg_x * _NSG * 32, ntg_y, 1),
              group_size=(_NSG * 32, 1, 1))
    return c


def self_check_ok():
    """Cached one-time numeric check: sg_matmul result must match a@b within tolerance."""
    global _self_check
    if _self_check is not None:
        return _self_check
    if not available():
        _self_check = False
        return False
    try:
        torch.manual_seed(0)
        a = (torch.randn(64, 256) * 0.5).to(torch.bfloat16).to("mps").contiguous()
        b = (torch.randn(256, 96) * 0.5).to(torch.bfloat16).to("mps").contiguous()
        ref = a.float() @ b.float()
        out = sg_matmul(a, b)
        rel = (out - ref).abs().max() / (ref.abs().max() + 1e-9)
        _self_check = bool(rel.item() < 5e-2)
        if not _self_check:
            print(f"{TAG} self-check failed (rel={rel.item():.4f}); sg_gemm disabled.")
    except Exception as e:
        print(f"{TAG} self-check raised; sg_gemm disabled ({e!r}).")
        _self_check = False
    return _self_check
