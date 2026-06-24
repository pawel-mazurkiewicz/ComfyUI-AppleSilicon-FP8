r"""GEMM bench harness: TF/s, GB/s, and parity for candidates vs MPS a@b.

Run: /Volumes/IMPERIAL\ SPACE/AI/ComfyUI/.venv/bin/python dev/bench_gemm.py
Not a pytest test — a dev measurement script. Shapes cover both regimes:
memory-bound (small M, large K/N — weight-dominated) and compute-bound (large M).
"""

import time

import torch

# (M, K, N) — FLUX/Krea2-ish Linear shapes. Memory-bound first, compute-bound last.
SHAPES = [
    (256, 3072, 3072),     # small batch, square-ish weight
    (256, 3072, 12288),    # MLP up-proj, weight-dominated
    (1024, 3072, 3072),
    (1024, 12288, 3072),   # MLP down-proj
    (4096, 3072, 3072),    # compute-bound
    (4096, 3072, 12288),   # compute-bound, big
]

WARMUP = 5
ITERS = 50


def _sync():
    torch.mps.synchronize()


def bench(fn, a, b, iters=ITERS):
    """Return (seconds_per_call, output). Times `iters` calls after warmup."""
    for _ in range(WARMUP):
        out = fn(a, b)
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(a, b)
    _sync()
    return (time.perf_counter() - t0) / iters, out


def tflops(m, k, n, sec):
    return (2.0 * m * k * n) / sec / 1e12


def gbps(m, k, n, sec, in_bytes):
    # Bytes read for operands (A: m*k, B: k*n) + C written (m*n, fp32=4B).
    moved = (m * k + k * n) * in_bytes + (m * n) * 4
    return moved / sec / 1e9


def rel_err(out, ref):
    return ((out.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-9)).item()


def candidates():
    """Map name -> (fn, operand_bytes_per_elem). Imported lazily so a broken
    backend doesn't abort the whole sweep."""
    cands = {"mps_bf16": (lambda a, b: a @ b, 2)}
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from _patches import na_gemm
        if na_gemm.available():
            cands["na_gemm_bf16"] = (lambda a, b: na_gemm.na_matmul(a, b), 2)
    except Exception as e:
        print(f"  (na_gemm unavailable: {e!r})")
    try:
        from _patches import sg_gemm
        if sg_gemm.available():
            cands["sg_gemm_bf16"] = (lambda a, b: sg_gemm.sg_matmul(a, b), 2)
    except Exception as e:
        print(f"  (sg_gemm unavailable: {e!r})")
    return cands


def main():
    if not torch.backends.mps.is_available():
        print("MPS not available; abort.")
        return
    cands = candidates()
    print(f"candidates: {list(cands)}\n")
    header = f"{'shape (M,K,N)':>22} | {'cand':>14} | {'ms':>8} | {'TF/s':>7} | {'GB/s':>7} | {'rel_err':>9}"
    print(header)
    print("-" * len(header))
    for (m, k, n) in SHAPES:
        torch.manual_seed(0)
        a = (torch.randn(m, k) * 0.3).to(torch.bfloat16).to("mps")
        b = (torch.randn(k, n) * 0.3).to(torch.bfloat16).to("mps")
        ref = a.float() @ b.float()
        for name, (fn, ob) in cands.items():
            try:
                sec, out = bench(fn, a, b)
                print(f"{str((m,k,n)):>22} | {name:>14} | {sec*1e3:8.3f} | "
                      f"{tflops(m,k,n,sec):7.1f} | {gbps(m,k,n,sec,ob):7.1f} | {rel_err(out,ref):9.2e}")
            except Exception as e:
                print(f"{str((m,k,n)):>22} | {name:>14} | FAILED: {e!r}")
        print()


if __name__ == "__main__":
    main()
