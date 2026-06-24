r"""fp8-native matmul2d: find the memory-bound -> compute-bound crossover + ragged parity.

Sweeps M (batch/token count) across both regimes for representative FLUX/Krea2 weight
shapes, reporting Y/X (fp8-native time / decode->MPS time) and parity. Y/X < 1 means
fp8-native wins; the crossover M tells the production shape heuristic where to route.
Also checks ragged (non-tile-multiple) shapes for correctness.

Run (ANNOUNCE GPU use first):
  ASFP8_FP8_EXT=1 /tmp/asfp8_venv/bin/python /tmp/asfp8_repo/dev/bench_fp8_crossover.py
  (run via the no-space symlinks; see fp8_ext/loader.py KNOWN SNAG.)
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fp8_ext.loader import load as load_ext
from _patches._common import decode_fp8

WEIGHTS = [(3072, 3072), (3072, 12288), (12288, 3072)]   # (K, N)
M_SWEEP = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
RAGGED = [(200, 130, 100), (257, 3071, 999), (1000, 4097, 513), (63, 65, 67)]


def bench(fn, iters=40, warmup=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def make_inputs(M, K, N):
    torch.manual_seed(1)
    a = (torch.randn(M, K) * 0.3).to(torch.bfloat16).to("mps")
    a_half = a.to(torch.half).contiguous()
    torch.manual_seed(0)
    w_fp8_cpu = (torch.randn(K, N) * 0.3).to(torch.float8_e4m3fn).contiguous()
    w_u8 = w_fp8_cpu.view(torch.uint8).to("mps").contiguous()
    w_fp8_mps = w_fp8_cpu.to("mps")
    return a, a_half, w_u8, w_fp8_mps


def main():
    if not torch.backends.mps.is_available():
        print("MPS not available; abort.")
        return
    mod = load_ext()
    if mod is None:
        print("extension unavailable; abort.")
        return
    mod.warmup()
    cooldown = float(os.environ.get("ASFP8_BENCH_COOLDOWN", "3.0"))

    print("\n=== ragged parity (rel must be < 5e-2) ===")
    all_ok = True
    for (M, K, N) in RAGGED:
        a, a_half, w_u8, w_fp8_mps = make_inputs(M, K, N)
        ref = a_half.float() @ decode_fp8(w_fp8_mps, torch.float32)
        out = mod.fp8_matmul2d(a_half, w_u8, N)
        rel = ((out - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        ok = rel < 5e-2
        all_ok &= ok
        print(f"  {str((M,K,N)):>20} rel={rel:.2e} {'OK' if ok else 'FAIL'}")
    print(f"  ragged parity: {'ALL OK' if all_ok else 'FAILURES'}")

    print("\n=== crossover sweep: Y/X (fp8-native / decode->MPS); <1 = fp8 wins ===")
    for (K, N) in WEIGHTS:
        print(f"\n  weight K={K} N={N}")
        print(f"    {'M':>6} | {'X ms':>8} | {'Y ms':>8} | {'Y/X':>5} | {'winner':>8}")
        for M in M_SWEEP:
            a, a_half, w_u8, w_fp8_mps = make_inputs(M, K, N)

            def pathX():
                wb = decode_fp8(w_fp8_mps, torch.bfloat16)
                return a.to(torch.bfloat16) @ wb

            def pathY():
                return mod.fp8_matmul2d(a_half, w_u8, N)

            tX = bench(pathX); time.sleep(cooldown)
            tY = bench(pathY); time.sleep(cooldown)
            r = tY / tX
            print(f"    {M:>6} | {tX*1e3:8.3f} | {tY*1e3:8.3f} | {r:5.2f} | "
                  f"{'fp8' if r < 1.0 else 'decode':>8}")


if __name__ == "__main__":
    main()
