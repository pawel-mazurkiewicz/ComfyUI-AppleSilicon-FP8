"""Probe: MPP matmul2d with packed-int4 weights on M5 (the facts:461 W4 probe).

Answers:
  1. Does it compile/run at Metal 4.1? (int4b_format gated like fp8)
  2. Nibble order: does MPP's int4b match comfy-kitchen's packing
     (low nibble = even column, signed two's complement)?
  3. Speed at Krea2/FLUX shapes: W4A8 (int8xint4b->int32) and
     W4A16 (bf16xint4b->float) vs MPS bf16 GEMM.

Run:
  "/Volumes/IMPERIAL SPACE/AI/ComfyUI/.venv/bin/python" dev/probe_int4_matmul2d.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("ASFP8_INT4_EXT", "1")

import torch  # noqa: E402

from _patches.int4_ext import loader  # noqa: E402

mod = loader.module()
if mod is None:
    print("BUILD FAILED — see loader output above")
    sys.exit(1)
print("extension built OK")

dev = "mps"
torch.manual_seed(0)


def pack_lo_even(q):  # comfy-kitchen order: low nibble = even column
    lo = q[..., 0::2].to(torch.int32) & 0x0F
    hi = q[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.uint8)


def pack_hi_even(q):  # swapped hypothesis
    lo = q[..., 1::2].to(torch.int32) & 0x0F
    hi = q[..., 0::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.uint8)


M, K, N = 256, 512, 384
a8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)  # full nibble range
ref = (a8.to(torch.int32).cpu() @ q4.to(torch.int32).cpu().T)

for name, packer in (("lo=even (kitchen)", pack_lo_even), ("hi=even (swapped)", pack_hi_even)):
    w4 = packer(q4).contiguous()
    C = mod.i8i4_matmul2d_nt(a8, w4, K, N).cpu()
    exact = torch.equal(C, ref)
    maxdiff = (C - ref).abs().max().item()
    print(f"W4A8  nibble order {name}: exact={exact} maxdiff={maxdiff}")

# W4A16 correctness (float accumulate; compare vs fp32 reference)
abf = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
ref_f = abf.float().cpu() @ q4.float().cpu().T
for name, packer in (("lo=even (kitchen)", pack_lo_even), ("hi=even (swapped)", pack_hi_even)):
    w4 = packer(q4).contiguous()
    C = mod.bf16i4_matmul2d_nt(abf, w4, K, N).cpu()
    rel = ((C - ref_f).abs().max() / ref_f.abs().max()).item()
    print(f"W4A16 nibble order {name}: max rel diff={rel:.3e}")


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


print("\nspeed @ Krea2-ish shapes (NOTE: contended if GPU busy)")
for (M, K, N) in ((4096, 6144, 6144), (4096, 6144, 24576), (4608, 4608, 4608)):
    a8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
    abf = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    wbf = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
    w4 = pack_lo_even(torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)).contiguous()

    t_bf = bench(lambda: torch.nn.functional.linear(abf, wbf))
    t_w4a8 = bench(lambda: mod.i8i4_matmul2d_nt(a8, w4, K, N))
    t_w4a16 = bench(lambda: mod.bf16i4_matmul2d_nt(abf, w4, K, N))
    tf = 2 * M * K * N / 1e12
    print(f"  M{M} K{K} N{N}: bf16 {t_bf:6.2f}ms ({tf / t_bf * 1e3:5.1f} TF/s) | "
          f"W4A8 {t_w4a8:6.2f}ms ({tf / t_w4a8 * 1e3:5.1f} TF/s) | "
          f"W4A16 {t_w4a16:6.2f}ms ({tf / t_w4a16 * 1e3:5.1f} TF/s)")
