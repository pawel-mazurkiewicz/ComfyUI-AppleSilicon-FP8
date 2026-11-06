"""Bench: where the W8A8 kernel path stops beating comfy's dequant paths.

Run from the repo root:
  PYTHONPATH=. "$PY" dev/bench_int8_crossover.py [--shapes K,N;K,N] [--tokens 512,...] [--rounds 5]

(a) _int8_linear_kernel, act cache cleared before each call (convrot, per-channel scales)
(a3) three weights on one x (q/k/v) sharing the cache, per-projection time
(b) bf16 weight dequantized once outside the loop, then F.linear
(c) per-call dequant (w.to(bf16) * scale) then F.linear
(eager) comfy_kitchen eager int8_linear, only with --eager: its _int_mm falls back to
    CPU on MPS (about 11 s per call at 512x3072x12288), so it is not a usable baseline
"""
import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from _patches import int8_linear_kernel_mps as patch

DEFAULT_SHAPES = "3072,12288;3840,10240;3072,12288"
DEFAULT_TOKENS = "512,1024,2048,4096,8192,16384"
ITERS = 20
COLS = ["a", "a3", "b", "c"]


def time_ms(fn, iters):
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def reset():
    patch._ACT_CACHE = None  # (a) must pay quantization on every call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=DEFAULT_SHAPES)
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--iters", type=int, default=ITERS)
    args = ap.parse_args()
    shapes = [tuple(int(v) for v in s.split(",")) for s in args.shapes.split(";")]
    tokens = [int(t) for t in args.tokens.split(",")]

    assert patch._ensure_kernel() is not None, "int8 extension unavailable"
    if args.eager:
        from comfy_kitchen.backends.eager.quantization import int8_linear as stock

    dev, bf = "mps", torch.bfloat16
    cols = COLS + (["eager"] if args.eager else [])
    print("K,N,M," + ",".join(f"{c}_ms" for c in cols))
    for K, N in shapes:
        g = torch.Generator().manual_seed(0)
        ws = [
            (torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8).to(dev),
             (torch.rand(N, generator=g) * 0.01 + 1e-3).to(dev))
            for _ in range(3)
        ]
        w, s = ws[0]
        w_bf = (w.float() * s[:, None]).to(bf)
        for M in tokens:
            x = torch.randn(M, K, generator=g).to(dev, bf)
            fns = {
                "a": lambda: (reset(), patch._int8_linear_kernel(x, w, s, None, bf, True, 256)),
                "a3": lambda: (reset(), [patch._int8_linear_kernel(x, wi, si, None, bf, True, 256)
                                         for wi, si in ws]),
                "b": lambda: F.linear(x, w_bf),
                "c": lambda: F.linear(x, (w.to(bf) * s[:, None].to(bf))),
            }
            if args.eager:
                fns["eager"] = lambda: stock(x, w, s, None, bf, True, 256)
            times = {k: [] for k in fns}
            for r in range(args.rounds):
                order = list(fns) if r % 2 == 0 else list(reversed(list(fns)))
                for k in order:
                    times[k].append(time_ms(fns[k], args.iters))
            med = {k: statistics.median(v) for k, v in times.items()}
            med["a3"] /= 3
            spread = max((max(v) - min(v)) / statistics.median(v) for v in times.values())
            print(f"{K},{N},{M}," + ",".join(f"{med[c]:.3f}" for c in cols)
                  + f"  # max spread {spread:.0%}", flush=True)
            del x


if __name__ == "__main__":
    main()
