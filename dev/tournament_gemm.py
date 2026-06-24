r"""GEMM kernel-variant tournament (dev R&D, M2 optimization).

Compiles candidate staged-GEMM kernels, parity-checks each (rel < 5e-2 on a ragged
shape), and benches ratio-to-MPS within a single run with a cooldown between
candidates (thermal-robust: the ratio to MPS measured in the same hot/cool state is
the signal, not absolute TF/s). MPS `a@b`, the un-staged `na_gemm`, and the M2a
baseline `sg_gemm` are reference rows. A variant that fails to compile or misses
parity is skipped with a printed reason — the sweep never aborts.

Run (ANNOUNCE GPU use first — do not run while ComfyUI is generating):
  /Volumes/IMPERIAL\ SPACE/AI/ComfyUI/.venv/bin/python dev/tournament_gemm.py

Diagnosed M2a culprits this targets:
  - small 64x64 output tile -> low arithmetic intensity  -> V2 (128-row tile, B reuse)
  - tmp-roundtrip + uncoalesced scalar store epilogue     -> V1 (direct simdgroup_store)
  - (double-buffering is a later add once a tile/store winner is picked)
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # dev/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

from bench_gemm import SHAPES, bench, tflops  # reuse harness helpers
from _patches import na_gemm, sg_gemm

COOLDOWN_S = float(os.environ.get("ASFP8_BENCH_COOLDOWN", "4.0"))

# --- Shared kernel body (preamble + staged load + compute), used by V1 -------
# V1 = M2a (half staging, 64x64 tile, 8 rows/simdgroup) with a DIRECT
# simdgroup_store to device C for full tiles and a scalar fallback only at ragged
# edges. Everything above the epilogue is identical to sg_gemm._GEMM_MSL.
_V1_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr int BM  = @BM@;
constant constexpr int BN  = @BN@;
constant constexpr int BK  = @BK@;
constant constexpr int NSG = @NSG@;

kernel void sgemm(
    device bfloat* A  [[buffer(0)]],
    device bfloat* B  [[buffer(1)]],
    device float*  C  [[buffer(2)]],
    device int*    SH [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int M = SH[0], N = SH[1], K = SH[2];
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;

    threadgroup half As[BM * BK];
    threadgroup half Bs[BK * BN];
    const int sr0 = int(sgid) * 8;

    simdgroup_float8x8 acc[BN / 8];
    for (int j = 0; j < BN / 8; ++j) acc[j] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    const int nthreads = NSG * 32;
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = int(tid); i < BM * BK; i += nthreads) {
            int r = i / BK, c = i % BK; int gr = m0 + r, gc = k0 + c;
            As[i] = half((gr < M && gc < K) ? float(A[ulong(gr) * K + gc]) : 0.0f);
        }
        for (int i = int(tid); i < BK * BN; i += nthreads) {
            int r = i / BN, c = i % BN; int gr = k0 + r, gc = n0 + c;
            Bs[i] = half((gr < K && gc < N) ? float(B[ulong(gr) * N + gc]) : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 Amat;
            simdgroup_load(Amat, &As[sr0 * BK + kk], ulong(BK));
            for (int j = 0; j < BN / 8; ++j) {
                simdgroup_half8x8 Bmat;
                simdgroup_load(Bmat, &Bs[kk * BN + j * 8], ulong(BN));
                simdgroup_multiply_accumulate(acc[j], Amat, Bmat, acc[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Direct store to device C for fully in-bounds 8x8 tiles; scalar fallback at edges.
    threadgroup float tmp[NSG * 64];
    for (int j = 0; j < BN / 8; ++j) {
        int gr0 = m0 + sr0, gc0 = n0 + j * 8;
        if (gr0 + 8 <= M && gc0 + 8 <= N) {
            simdgroup_store(acc[j], C + ulong(gr0) * N + gc0, ulong(N));
        } else {
            simdgroup_store(acc[j], &tmp[int(sgid) * 64], ulong(8));
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int e = int(lane); e < 64; e += 32) {
                int r = e / 8, c = e % 8; int gr = gr0 + r, gc = gc0 + c;
                if (gr < M && gc < N) C[ulong(gr) * N + gc] = tmp[int(sgid) * 64 + r * 8 + c];
            }
        }
    }
}
"""

# --- V2: 128-row output tile, each simdgroup owns RT=BM/(NSG*8) row sub-tiles ---
# Each loaded B 8x8 sub-tile is reused across RT row-tiles (RT=2 for BM=128,NSG=8),
# doubling A-row reuse of staged B -> higher arithmetic intensity. Direct store.
_V2_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr int BM  = @BM@;
constant constexpr int BN  = @BN@;
constant constexpr int BK  = @BK@;
constant constexpr int NSG = @NSG@;
constant constexpr int RT  = BM / (NSG * 8);   // row-tiles per simdgroup

kernel void sgemm(
    device bfloat* A  [[buffer(0)]],
    device bfloat* B  [[buffer(1)]],
    device float*  C  [[buffer(2)]],
    device int*    SH [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int M = SH[0], N = SH[1], K = SH[2];
    const int m0 = int(tgid.x) * BM;
    const int n0 = int(tgid.y) * BN;

    threadgroup half As[BM * BK];
    threadgroup half Bs[BK * BN];
    const int sr0 = int(sgid) * (RT * 8);   // first output row for this simdgroup

    simdgroup_float8x8 acc[RT][BN / 8];
    for (int ri = 0; ri < RT; ++ri)
        for (int j = 0; j < BN / 8; ++j) acc[ri][j] = make_filled_simdgroup_matrix<float,8,8>(0.0f);

    const int nthreads = NSG * 32;
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = int(tid); i < BM * BK; i += nthreads) {
            int r = i / BK, c = i % BK; int gr = m0 + r, gc = k0 + c;
            As[i] = half((gr < M && gc < K) ? float(A[ulong(gr) * K + gc]) : 0.0f);
        }
        for (int i = int(tid); i < BK * BN; i += nthreads) {
            int r = i / BN, c = i % BN; int gr = k0 + r, gc = n0 + c;
            Bs[i] = half((gr < K && gc < N) ? float(B[ulong(gr) * N + gc]) : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 Amat[RT];
            for (int ri = 0; ri < RT; ++ri)
                simdgroup_load(Amat[ri], &As[(sr0 + ri * 8) * BK + kk], ulong(BK));
            for (int j = 0; j < BN / 8; ++j) {
                simdgroup_half8x8 Bmat;
                simdgroup_load(Bmat, &Bs[kk * BN + j * 8], ulong(BN));
                for (int ri = 0; ri < RT; ++ri)
                    simdgroup_multiply_accumulate(acc[ri][j], Amat[ri], Bmat, acc[ri][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float tmp[NSG * 64];
    for (int ri = 0; ri < RT; ++ri) {
        for (int j = 0; j < BN / 8; ++j) {
            int gr0 = m0 + sr0 + ri * 8, gc0 = n0 + j * 8;
            if (gr0 + 8 <= M && gc0 + 8 <= N) {
                simdgroup_store(acc[ri][j], C + ulong(gr0) * N + gc0, ulong(N));
            } else {
                simdgroup_store(acc[ri][j], &tmp[int(sgid) * 64], ulong(8));
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (int e = int(lane); e < 64; e += 32) {
                    int r = e / 8, c = e % 8; int gr = gr0 + r, gc = gc0 + c;
                    if (gr < M && gc < N) C[ulong(gr) * N + gc] = tmp[int(sgid) * 64 + r * 8 + c];
                }
            }
        }
    }
}
"""

# Each variant: name -> (msl, BM, BN, BK, NSG). BM must equal 8*NSG*RT (RT>=1).
VARIANTS = {
    "v1_directstore": (_V1_MSL, 64, 64, 32, 8),
    "v2_tile128x64":  (_V2_MSL, 128, 64, 32, 8),
    "v2_tile128x128": (_V2_MSL, 128, 128, 32, 8),
}


def compile_variant(msl, bm, bn, bk, nsg):
    src = (msl.replace("@BM@", str(bm)).replace("@BN@", str(bn))
              .replace("@BK@", str(bk)).replace("@NSG@", str(nsg)))
    return torch.mps.compile_shader(src)


def make_matmul(lib, bm, bn, nsg):
    def fn(a, b):
        a = a.contiguous(); b = b.contiguous()
        M, K = a.shape; K2, N = b.shape
        assert K == K2
        c = torch.zeros(M, N, device="mps", dtype=torch.float32)
        sh = torch.tensor([M, N, K], dtype=torch.int32, device="mps")
        ntg_x = -(-M // bm); ntg_y = -(-N // bn)
        lib.sgemm(a, b, c, sh, threads=(ntg_x * nsg * 32, ntg_y, 1), group_size=(nsg * 32, 1, 1))
        return c
    return fn


def parity_ok(fn):
    """Ragged + exact-multiple shapes must match a@b within 5e-2."""
    for (m, k, n) in [(200, 130, 100), (128, 256, 192), (64, 64, 64)]:
        torch.manual_seed(0)
        a = (torch.randn(m, k) * 0.3).to(torch.bfloat16).to("mps")
        b = (torch.randn(k, n) * 0.3).to(torch.bfloat16).to("mps")
        ref = a.float() @ b.float()
        out = fn(a, b)
        rel = ((out - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        if rel >= 5e-2:
            return False, rel, (m, k, n)
    return True, 0.0, None


def build_candidates():
    cands = {"mps_bf16": lambda a, b: a @ b}
    if na_gemm.available():
        cands["na_gemm"] = lambda a, b: na_gemm.na_matmul(a, b)
    if sg_gemm.available():
        cands["sg_gemm_m2a"] = lambda a, b: sg_gemm.sg_matmul(a, b)
    for name, (msl, bm, bn, bk, nsg) in VARIANTS.items():
        try:
            lib = compile_variant(msl, bm, bn, bk, nsg)
        except Exception as e:
            print(f"  [skip] {name}: compile failed: {e!r}")
            continue
        fn = make_matmul(lib, bm, bn, nsg)
        ok, rel, shape = parity_ok(fn)
        if not ok:
            print(f"  [skip] {name}: parity rel={rel:.4f} at {shape}")
            continue
        cands[name] = fn
        print(f"  [ok]   {name}: compiled + parity passed")
    return cands


def main():
    if not torch.backends.mps.is_available():
        print("MPS not available; abort.")
        return
    print(f"cooldown {COOLDOWN_S}s between candidates\n")
    cands = build_candidates()
    print(f"\ncandidates: {list(cands)}\n")
    header = f"{'shape (M,K,N)':>22} | {'cand':>16} | {'TF/s':>7} | {'x MPS':>6}"
    for (m, k, n) in SHAPES:
        torch.manual_seed(0)
        a = (torch.randn(m, k) * 0.3).to(torch.bfloat16).to("mps")
        b = (torch.randn(k, n) * 0.3).to(torch.bfloat16).to("mps")
        print(header)
        print("-" * len(header))
        mps_tf = None
        for name, fn in cands.items():
            try:
                sec, _ = bench(fn, a, b)
                tf = tflops(m, k, n, sec)
                if name == "mps_bf16":
                    mps_tf = tf
                ratio = (tf / mps_tf) if mps_tf else float("nan")
                print(f"{str((m,k,n)):>22} | {name:>16} | {tf:7.1f} | {ratio:6.2f}")
            except Exception as e:
                print(f"{str((m,k,n)):>22} | {name:>16} | FAILED: {e!r}")
            time.sleep(COOLDOWN_S)
        print()


if __name__ == "__main__":
    main()
