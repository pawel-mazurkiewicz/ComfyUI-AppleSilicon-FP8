"""Head-to-head: bf16 vs int8xint8 vs W4A8 (int8xint4b) in ONE process.

Settles whether int4 can beat int8 on M5 at Krea2/FLUX shapes, or whether the
weight-byte saving buys nothing because the GEMM is compute-bound.

Run:
  "/Volumes/IMPERIAL SPACE/AI/ComfyUI/.venv/bin/python" dev/probe_int4_vs_int8.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("ASFP8_INT4_EXT", "1")
os.environ.setdefault("ASFP8_INT8_EXT", "1")

import torch  # noqa: E402

from _patches.int4_ext import loader as i4_loader  # noqa: E402
from _patches.int8_ext import loader as i8_loader  # noqa: E402

m4 = i4_loader.module()
m8 = i8_loader.module()
if m4 is None or m8 is None:
    print(f"BUILD FAILED int4={m4 is not None} int8={m8 is not None}")
    sys.exit(1)
print("both extensions built OK")

dev = "mps"
torch.manual_seed(0)


def pack_lo_even(q):
    lo = q[..., 0::2].to(torch.int32) & 0x0F
    hi = q[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.uint8)


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


print("\nsame process, same shapes — GEMM throughput only (no rotation/quant):")
for (M, K, N) in ((4096, 6144, 6144), (4096, 6144, 24576), (4608, 4608, 4608)):
    a8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
    abf = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    wbf = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
    w8 = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
    w4 = pack_lo_even(torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)).contiguous()

    t_bf = bench(lambda: torch.nn.functional.linear(abf, wbf))
    t_i8 = bench(lambda: m8.i8_matmul2d_nt(a8, w8))
    t_w4a8 = bench(lambda: m4.i8i4_matmul2d_nt(a8, w4, K, N))
    tf = 2 * M * K * N / 1e12
    print(f"  M{M} K{K} N{N}:")
    print(f"    bf16      {t_bf:6.2f}ms ({tf / t_bf * 1e3:5.1f} TF/s)  1.00x")
    print(f"    int8xint8 {t_i8:6.2f}ms ({tf / t_i8 * 1e3:5.1f} TF/s)  {t_bf / t_i8:.2f}x")
    print(f"    W4A8      {t_w4a8:6.2f}ms ({tf / t_w4a8 * 1e3:5.1f} TF/s)  {t_bf / t_w4a8:.2f}x"
          f"   |  vs int8: {t_i8 / t_w4a8:.2f}x")
