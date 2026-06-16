"""Tier-1 check: bf16-decode _scaled_mm vs the old f32-decode path on MPS.

    python dev/bench_scaled_mm_dtype.py

Confirms FP8->bf16 decode + bf16 matmul beats FP8->f32 decode + f32 matmul on the
matrix units. FLUX-ish shapes. Apple Silicon only.
"""
import time

import torch


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
        print("MPS unavailable; skipping"); return
    for (M, K, N) in [(4096, 4096, 4096), (1024, 4096, 12288), (8192, 3072, 3072)]:
        a = torch.randn(M, K, device="mps")
        b = torch.randn(K, N, device="mps")
        a16, b16 = a.to(torch.bfloat16), b.to(torch.bfloat16)
        flops = 2 * M * N * K
        tf32 = bench(lambda: a @ b)
        tbf = bench(lambda: a16 @ b16)
        print(f"M{M} K{K} N{N}: f32 {flops/tf32/1e12:5.1f} TF/s | "
              f"bf16 {flops/tbf/1e12:5.1f} TF/s | speedup {tf32/tbf:.2f}x")


if __name__ == "__main__":
    main()
