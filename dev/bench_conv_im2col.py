# dev/bench_conv_im2col.py
import time

import torch

from _patches.conv_im2col_mps import conv_im2col


def bench(fn, it=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / it * 1e3


def verify(ours, x, w, **kw):
    """Correctness gate BEFORE timing: compare to fp32 reference; report max-diff."""
    ref = torch.nn.functional.conv2d(x.float(), w.float(), **kw) if w.dim() == 4 \
        else torch.nn.functional.conv3d(x.float(), w.float(), **kw)
    got = ours().float()
    maxdiff = (got - ref).abs().max().item()
    ok = maxdiff < 2e-1
    print(f"  correctness: max|diff|={maxdiff:.4g} -> {'OK' if ok else 'FAIL'}")
    return ok


def main():
    x = torch.randn(1, 256, 512, 512, device="mps", dtype=torch.float16)
    w = torch.randn(256, 256, 3, 3, device="mps", dtype=torch.float16)
    if not verify(lambda: conv_im2col(x, w, None, 1, 1), x, w, padding=1):
        print("DECISION: correctness FAIL -> do not ship regardless of speed")
        return
    torch.mps.empty_cache()
    ours = bench(lambda: conv_im2col(x, w, None, 1, 1))
    stock = bench(lambda: torch.nn.functional.conv2d(x, w, padding=1))
    print(f"conv2d  im2col={ours:.3f}ms  stock={stock:.3f}ms  speedup={stock / ours:.2f}x")
    print(f"  current_allocated (NOT peak): {torch.mps.current_allocated_memory() / 1024**3:.2f}GiB")

    # --- conv3d (SeedVR2-ish), appended in Task B.6 ---
    x3 = torch.randn(1, 128, 5, 256, 256, device="mps", dtype=torch.float16)
    w3 = torch.randn(128, 128, 3, 3, 3, device="mps", dtype=torch.float16)
    if not verify(lambda: conv_im2col(x3, w3, None, 1, 1), x3, w3, padding=1):
        print("DECISION: conv3d correctness FAIL -> do not ship regardless of speed")
        return
    torch.mps.empty_cache()
    o3 = bench(lambda: conv_im2col(x3, w3, None, 1, 1))
    cur = torch.mps.current_allocated_memory() / 1024**3  # current, NOT peak
    s3 = bench(lambda: torch.nn.functional.conv3d(x3, w3, padding=1))
    print(f"conv3d  im2col={o3:.3f}ms  stock={s3:.3f}ms  speedup={s3 / o3:.2f}x  "
          f"current_alloc_after_ours={cur:.2f}GiB (NOT peak)")


if __name__ == "__main__":
    main()
