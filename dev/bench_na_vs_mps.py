"""Does the vendored NA matmul2d beat MPS's own bf16 GEMM? FLUX-ish shapes.

    python dev/bench_na_vs_mps.py
"""
import sys
import time

import torch

sys.path.insert(0, ".")
from _patches import na_gemm


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
    if not torch.backends.mps.is_available() or not na_gemm.available():
        print("MPS/NA unavailable; skipping"); return
    for (M, K, N) in [(4096, 4096, 4096), (1024, 4096, 12288), (8192, 3072, 3072)]:
        a = torch.randn(M, K, device="mps", dtype=torch.bfloat16).contiguous()
        b = torch.randn(K, N, device="mps", dtype=torch.bfloat16).contiguous()
        flops = 2 * M * N * K
        tmps = bench(lambda: a @ b)
        tna = bench(lambda: na_gemm.na_matmul(a, b))
        print(f"M{M} K{K} N{N}: MPS {flops/tmps/1e12:5.1f} TF/s | "
              f"NA {flops/tna/1e12:5.1f} TF/s | NA/MPS {tmps/tna:.2f}x")


if __name__ == "__main__":
    main()
