"""Bench: per-channel W8A8 epilogue, chunked int32 rescale vs i8_matmul2d_nt_fused_cs.

Run from the repo root: PYTHONPATH=. "$PY" dev/bench_int8_epilogue.py
"""
import statistics
import time

import torch

from _patches.int8_ext import loader

SHAPES = [(4096, 3072, 3072), (4096, 3072, 12288), (4096, 12288, 3072)]
ITERS = 20
ROUNDS = 5


def chunked(mod, x8, xs, w, ws):
    result = mod.i8_matmul2d_nt(x8, w)
    m, n = result.shape
    chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))
    parts = []
    for i in range(0, m, chunk_size):
        end_i = min(i + chunk_size, m)
        parts.append((result[i:end_i].float() * (ws * xs[i:end_i])).to(torch.bfloat16))
    return torch.cat(parts, dim=0)


def fused_cs(mod, x8, xs, w, ws):
    return mod.i8_matmul2d_nt_fused_cs(x8, w, xs.reshape(-1), ws, None, 0)


def time_ms(fn):
    for _ in range(ITERS):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / ITERS * 1e3


def main():
    mod = loader.module()
    assert mod is not None and hasattr(mod, "i8_matmul2d_nt_fused_cs")
    mod.warmup()
    print(f"{'M,K,N':>18} | {'chunked ms':>10} | {'fused_cs ms':>11} | speedup")
    for M, K, N in SHAPES:
        g = torch.Generator().manual_seed(0)
        x8 = torch.randint(-128, 128, (M, K), generator=g, dtype=torch.int8).to("mps")
        w = torch.randint(-128, 128, (N, K), generator=g, dtype=torch.int8).to("mps")
        xs = (torch.rand(M, 1, generator=g) * 0.01 + 1e-4).to("mps")
        ws = (torch.rand(N, generator=g) * 0.01 + 1e-3).to("mps")
        fns = {
            "chunked": lambda: chunked(mod, x8, xs, w, ws),
            "fused_cs": lambda: fused_cs(mod, x8, xs, w, ws),
        }
        assert torch.equal(fns["chunked"](), fns["fused_cs"]()), "fused_cs != chunked"
        times = {k: [] for k in fns}
        for r in range(ROUNDS):
            order = list(fns) if r % 2 == 0 else list(reversed(fns))
            for k in order:
                times[k].append(time_ms(fns[k]))
        c, f = statistics.median(times["chunked"]), statistics.median(times["fused_cs"])
        print(f"{str((M, K, N)):>18} | {c:10.3f} | {f:11.3f} | {c / f:5.2f}x")


if __name__ == "__main__":
    main()
