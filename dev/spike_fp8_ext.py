r"""DECISIVE spike: fp8-native matmul2d (Metal 4.1 ObjC++ extension) vs decode->MPS.

Stages: (1) build the extension, (2) warmup = compile the Metal 4.1 fp8 kernel
(proves the gated type is reachable via newLibraryWithSource:options:), (3) parity
vs a_half @ decode_fp8(W).float(), (4) bench fp8-native vs decode->bf16->MPS in the
memory-bound regime (the only place the bandwidth win can show).

Run (ANNOUNCE GPU use first):
  ASFP8_FP8_EXT=1 /Volumes/IMPERIAL\ SPACE/AI/ComfyUI/.venv/bin/python dev/spike_fp8_ext.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # dev/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

from fp8_ext.loader import load as load_ext
from _patches._common import decode_fp8


def bench(fn, iters=50, warmup=5):
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

    print("=== stage 1: build extension ===")
    mod = load_ext()
    if mod is None:
        print("extension unavailable; abort (see messages above).")
        return
    print("  built OK")

    print("=== stage 2: warmup (compile Metal 4.1 fp8 kernel) ===")
    try:
        mod.warmup()
        print("  fp8 kernel compiled at Metal 4.1 — gated type reachable via the shim.")
    except Exception as e:
        print(f"  FAILED: {e!r}")
        return

    cooldown = float(os.environ.get("ASFP8_BENCH_COOLDOWN", "4.0"))
    SHAPES = [(256, 3072, 3072), (256, 3072, 12288), (1024, 12288, 3072)]

    print("\n=== stage 3+4: parity + bench (X = decode->MPS, Y = fp8-native) ===")
    hdr = f"{'shape (M,K,N)':>22} | {'pathX ms':>9} | {'pathY ms':>9} | {'Y/X':>5} | {'parity rel':>10}"
    print(hdr); print("-" * len(hdr))
    for (M, K, N) in SHAPES:
        torch.manual_seed(1)
        a = (torch.randn(M, K) * 0.3).to(torch.bfloat16).to("mps")
        a_half = a.to(torch.half).contiguous()
        torch.manual_seed(0)
        w_fp8_cpu = (torch.randn(K, N) * 0.3).to(torch.float8_e4m3fn).contiguous()
        w_u8 = w_fp8_cpu.view(torch.uint8).to("mps").contiguous()
        w_fp8_mps = w_fp8_cpu.to("mps")

        w_ref = decode_fp8(w_fp8_mps, torch.float32)
        ref = a_half.float() @ w_ref

        try:
            outY = mod.fp8_matmul2d(a_half, w_u8, N)
            rel = ((outY - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        except Exception as e:
            print(f"{str((M,K,N)):>22} | fp8 run FAILED: {e!r}")
            continue

        def pathX():
            wb = decode_fp8(w_fp8_mps, torch.bfloat16)
            return a.to(torch.bfloat16) @ wb

        def pathY():
            return mod.fp8_matmul2d(a_half, w_u8, N)

        tX = bench(pathX); time.sleep(cooldown)
        tY = bench(pathY); time.sleep(cooldown)
        print(f"{str((M,K,N)):>22} | {tX*1e3:9.3f} | {tY*1e3:9.3f} | {tY/tX:5.2f} | {rel:10.2e}")

    print("\n(Y/X < 1.0 => fp8-native matmul2d beats decode->MPS. parity rel should be "
          "~fp8 quant noise, < 5e-2.)")


if __name__ == "__main__":
    main()
