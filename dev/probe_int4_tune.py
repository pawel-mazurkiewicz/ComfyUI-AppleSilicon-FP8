"""Sweep int4b matmul2d tile configs (dv_i4 vs tg_i4) and compare to the int8 kernel.

Tests the claim that our 84 TF/s int4 number was an untuned-kernel artifact, not
an M5 int4 ceiling. Every config is verified bit-exact against the int32 reference
before it is timed — a fast wrong kernel is not a result.

Run:
  "/Volumes/IMPERIAL SPACE/AI/ComfyUI/.venv/bin/python" dev/probe_int4_tune.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("ASFP8_INT4_EXT", "1")
os.environ.setdefault("ASFP8_INT8_EXT", "1")

import torch  # noqa: E402
from torch.utils.cpp_extension import load as cpp_load  # noqa: E402

from _patches.int4_ext import loader as i4_loader  # noqa: E402
from _patches.int8_ext import loader as i8_loader  # noqa: E402

_NOSPACE = "/tmp/asfp8_build"


def load_tune():
    import torch.utils.cpp_extension as cpp
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_patches", "int4_ext")
    src = os.path.abspath(os.path.join(here, "int4_tune.mm"))
    build_dir = os.path.join(_NOSPACE, "int4_tune")
    os.makedirs(build_dir, exist_ok=True)
    saved = cpp.TORCH_LIB_PATH
    try:
        cpp.TORCH_LIB_PATH = i4_loader._nospace_torch_lib()
        return cpp_load(name="asfp8_int4_tune", sources=[src],
                        extra_cflags=["-std=c++17", "-ObjC++"],
                        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
                        build_directory=build_dir, verbose=False)
    finally:
        cpp.TORCH_LIB_PATH = saved


mt = load_tune()
m8 = i8_loader.module()
if mt is None or m8 is None:
    print("BUILD FAILED")
    sys.exit(1)
print("tune + int8 extensions built OK\n")

dev = "mps"
torch.manual_seed(0)


def pack_lo_even(q):
    lo = q[..., 0::2].to(torch.int32) & 0x0F
    hi = q[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.uint8)


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


# (BM, BN, NSG, BK, use_tg)
CONFIGS = [
    (64, 64, 4, 0, 0),      # current shipped probe — the 84 TF/s baseline
    (128, 64, 4, 0, 0),
    (128, 128, 4, 0, 0),
    (256, 64, 4, 0, 0),
    (128, 128, 8, 0, 0),
    (256, 128, 8, 0, 0),
    (64, 64, 4, 128, 1),    # tg_i4 staging
    (128, 64, 4, 128, 1),
    (128, 128, 4, 128, 1),
    (128, 128, 4, 256, 1),
    (256, 128, 8, 128, 1),
    (256, 128, 8, 256, 1),
]

M, K, N = 4096, 6144, 6144
a8 = torch.randint(-127, 128, (M, K), dtype=torch.int8, device=dev)
q4 = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)
w4 = pack_lo_even(q4).contiguous()
w8 = torch.randint(-127, 128, (N, K), dtype=torch.int8, device=dev)
abf = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
wbf = torch.randn(N, K, dtype=torch.bfloat16, device=dev)

print(f"shape M{M} K{K} N{N}\ncomputing int32 reference on CPU...")
ref = (a8.to(torch.int32).cpu() @ q4.to(torch.int32).cpu().T)
tf = 2 * M * K * N / 1e12

t_bf = bench(lambda: torch.nn.functional.linear(abf, wbf))
t_i8 = bench(lambda: m8.i8_matmul2d_nt(a8, w8))
print(f"\nbaselines:")
print(f"  bf16       {t_bf:6.2f}ms ({tf / t_bf * 1e3:6.1f} TF/s)")
print(f"  int8xint8  {t_i8:6.2f}ms ({tf / t_i8 * 1e3:6.1f} TF/s)   <- target to beat\n")

print(f"{'BM':>4} {'BN':>4} {'NSG':>4} {'BK':>4} {'mode':>5} {'exact':>6} {'ms':>8} {'TF/s':>7} {'vs int8':>8}")
print("-" * 62)
best = None
for (BM, BN, NSG, BK, tg) in CONFIGS:
    mode = "tg_i4" if tg else "dv_i4"
    try:
        C = mt.i8i4_tuned(a8, w4, K, N, BM, BN, NSG, BK if BK else 2, tg)
        exact = torch.equal(C.cpu(), ref)
        if not exact:
            print(f"{BM:>4} {BN:>4} {NSG:>4} {BK:>4} {mode:>5} {'WRONG':>6}  (skipped)")
            continue
        t = bench(lambda: mt.i8i4_tuned(a8, w4, K, N, BM, BN, NSG, BK if BK else 2, tg))
        tfs = tf / t * 1e3
        print(f"{BM:>4} {BN:>4} {NSG:>4} {BK:>4} {mode:>5} {'yes':>6} {t:>8.2f} {tfs:>7.1f} {t_i8 / t:>7.2f}x")
        if best is None or t < best[0]:
            best = (t, BM, BN, NSG, BK, mode)
    except Exception as e:
        msg = str(e).split("\n")[0][:70]
        print(f"{BM:>4} {BN:>4} {NSG:>4} {BK:>4} {mode:>5} {'ERR':>6}  {msg}")

if best:
    t, BM, BN, NSG, BK, mode = best
    print(f"\nbest int4: BM{BM} BN{BN} NSG{NSG} BK{BK} {mode} -> {t:.2f}ms ({tf / t * 1e3:.1f} TF/s)")
    print(f"  vs int8 kernel: {t_i8 / t:.2f}x   |  vs bf16: {t_bf / t:.2f}x")
    print(f"  vs shipped int4 probe: see BM64 BN64 NSG4 dv_i4 row above")
