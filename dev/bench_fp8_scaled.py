r"""Bench the scaled-fp8 win: fp8xfp8 matmul2d + scales  vs  decode-both->bf16->matmul->scales.

This is the ACTUAL path ComfyUI scaled-fp8 checkpoints (Krea2/FLUX fp8_scaled) take:
both operands are fp8 (activation dynamically quantized + weight fp8), matmul via
_scaled_mm, then per-row/col scales. Decides whether hooking patch #3 (_scaled_mm) with
an fp8-native kernel is worth it.

  pathX (today): decode_fp8(a)->bf16, decode_fp8(W)->bf16, bf16 (a @ Wᵀ), * scale_a * scale_b
  pathY (probe): fp8xfp8 matmul2d -> f32 (unscaled),                      * scale_a * scale_b

Run (ANNOUNCE GPU; via no-space symlinks per loader KNOWN SNAG):
  ASFP8_FP8_EXT=1 /tmp/asfp8_venv/bin/python /tmp/asfp8_repo/dev/bench_fp8_scaled.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _patches.fp8_ext.loader import module as load_ext
from _patches._common import decode_fp8

WEIGHTS = [(3072, 12288), (12288, 3072), (3072, 3072)]   # (K, N)
M_SWEEP = [1024, 4096]


def bench(fn, iters=40, warmup=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    if not torch.backends.mps.is_available():
        print("MPS not available; abort.")
        return
    mod = load_ext()
    if mod is None:
        print("extension unavailable; abort.")
        return
    mod.warmup()
    cooldown = float(os.environ.get("ASFP8_BENCH_COOLDOWN", "2.0"))

    hdr = f"{'shape (M,K,N)':>22} | {'pathX ms':>9} | {'pathY ms':>9} | {'Y/X':>5} | {'parity rel':>10}"
    print(hdr); print("-" * len(hdr))
    for (K, N) in WEIGHTS:
        for M in M_SWEEP:
            torch.manual_seed(1)
            a_fp8 = (torch.randn(M, K) * 0.3).to(torch.float8_e4m3fn).contiguous()
            torch.manual_seed(0)
            w_fp8 = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn).contiguous()
            a_u8 = a_fp8.view(torch.uint8).to("mps").contiguous()
            w_u8 = w_fp8.view(torch.uint8).to("mps").contiguous()
            a_mps, w_mps = a_fp8.to("mps"), w_fp8.to("mps")
            scale_a = (torch.rand(M, 1) * 0.5 + 0.5).to("mps")
            scale_b = (torch.rand(1, N) * 0.5 + 0.5).to("mps")

            ref = (decode_fp8(a_mps, torch.float32) @ decode_fp8(w_mps, torch.float32).t()) * scale_a * scale_b

            raw = mod.fp8fp8_matmul2d_nt(a_u8, w_u8, K, N)
            outY = raw * scale_a * scale_b
            rel = ((outY - ref).abs().max() / (ref.abs().max() + 1e-9)).item()

            def pathX():
                ab = decode_fp8(a_mps, torch.bfloat16)
                wb = decode_fp8(w_mps, torch.bfloat16)
                return (ab @ wb.t()).float() * scale_a * scale_b

            def pathY():
                return mod.fp8fp8_matmul2d_nt(a_u8, w_u8, K, N) * scale_a * scale_b

            tX = bench(pathX); time.sleep(cooldown)
            tY = bench(pathY); time.sleep(cooldown)
            print(f"{str((M,K,N)):>22} | {tX*1e3:9.3f} | {tY*1e3:9.3f} | {tY/tX:5.2f} | {rel:10.2e}")
        print()

    print("(Y/X < 1 => fp8-native scaled path beats decode->bf16. parity rel < 5e-2 expected.)")


if __name__ == "__main__":
    main()
