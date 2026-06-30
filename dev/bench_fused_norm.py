# dev/bench_fused_norm.py
"""Bench the fused kernel vs the separate-op torch composition at a DiT adaLN-tail shape.
Reports first-call (compile) ms, steady-state ms, speedup, and an approximate achieved bandwidth.
PASS = fused matches separate (allclose) AND ran on the kernel path AND fused < separate."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from _patches import fused_norm_mps as m
from _patches.fused_norm_mps import fused_rmsnorm_modulate

assert torch.backends.mps.is_available(), "bench needs an MPS device"

rows, dim = 1 << 20, 1536
dt = torch.float16
x = torch.randn(rows, dim, device="mps", dtype=dt)
w = torch.randn(dim, device="mps", dtype=dt)
sc = torch.randn(dim, device="mps", dtype=dt)
sh = torch.randn(dim, device="mps", dtype=dt)
res = torch.randn(rows, dim, device="mps", dtype=dt)


def separate():
    h = torch.nn.functional.rms_norm(x, (dim,), w, 1e-6)
    return res + h * (1.0 + sc) + sh


def fused():
    return fused_rmsnorm_modulate(x, w, 1e-6, sc, sh, res)


# ---- correctness + kernel-path proof BEFORE timing ----
out_f = fused()
assert m._last_backend == "kernel", "fused() fell back to torch composition; benchmark would be a lie"
out_s = separate()
torch.mps.synchronize()
maxdiff = (out_f.float() - out_s.float()).abs().max().item()
assert torch.allclose(out_f.float(), out_s.float(), atol=5e-2, rtol=5e-2), \
    f"CORRECTNESS FAIL: max abs diff {maxdiff:.4g} — do not trust the timing"
print(f"correctness OK (max abs diff {maxdiff:.4g}, backend={m._last_backend})")

# ---- first-call (compile) cost, measured on a fresh dtype lib ----
m._libs.clear()
torch.mps.synchronize()
t0 = time.perf_counter()
fused()
torch.mps.synchronize()
compile_ms = (time.perf_counter() - t0) * 1e3
print(f"first-call (compile_shader) latency: {compile_ms:.1f} ms")


def bench(fn, it=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / it * 1e3


sep_ms = bench(separate)
fus_ms = bench(fused)
assert m._last_backend == "kernel", "fused path changed to fallback during timing"
bytes_fused = 3 * rows * dim * x.element_size()  # read x + read residual + write out
gbps = bytes_fused / (fus_ms * 1e-3) / 1e9
print(f"rows={rows} dim={dim} dtype={dt}")
print(f"separate={sep_ms:.3f}ms  fused={fus_ms:.3f}ms  speedup={sep_ms/fus_ms:.2f}x  "
      f"fused~{gbps:.0f} GB/s (BW estimate only; no measured roofline)")
assert fus_ms < sep_ms, "REGRESSION: fused slower than separate ops"
