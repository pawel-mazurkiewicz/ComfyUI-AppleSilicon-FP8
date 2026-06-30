"""Bench: fused SiLU epilogue vs (fused-no-act kernel + torch.silu)."""
import os, time, torch
os.environ.setdefault("ASFP8_INT8_EXT", "1")
from _patches.int8_ext import loader
mod = loader.module(); assert mod is not None; mod.warmup()

dev = "mps"
M, K, N = 4096, 1536, 6144
x8 = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=dev)
w  = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=dev)
rs = (torch.rand(M, device=dev) * 0.01 + 0.001).float().contiguous()
b  = torch.randn(N, device=dev, dtype=torch.bfloat16)

def fused():     return mod.i8_matmul2d_nt_fused(x8, w, rs, b, 1)         # act=silu
def separate():  return torch.nn.functional.silu(mod.i8_matmul2d_nt_fused(x8, w, rs, b, 0))

# Correctness FIRST (review MAJOR #8): a fast-but-wrong kernel must fail before any timing.
# Tolerance is one bf16 ulp (rtol=8e-3) + atol=2e-3 near zero — see D-results.md / the
# test rationale: 2e-3 rtol is below bf16 precision and unsatisfiable at large magnitude.
of, os_ = fused(), separate(); torch.mps.synchronize()
md = (of.float() - os_.float()).abs().max().item()
print(f"correctness: max|d|={md:.4g}")
assert torch.allclose(of, os_, atol=2e-3, rtol=8e-3), f"CORRECTNESS FAIL: max|d|={md:.4g}"

def bench(fn, it=50):
    fn(); torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.mps.synchronize(); return (time.perf_counter() - t) / it * 1e3

f, s = bench(fused), bench(separate)
print(f"M={M} K={K} N={N}: separate={s:.3f}ms fused={f:.3f}ms speedup={s/f:.2f}x")
assert f <= s * 1.02, "fused must not be slower than separate"
