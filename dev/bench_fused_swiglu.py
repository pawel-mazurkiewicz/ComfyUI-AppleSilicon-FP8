"""Bench: fused SwiGLU kernel vs (2 fused-no-act kernels + torch silu + mul)."""
import os, time, torch
os.environ.setdefault("ASFP8_INT8_EXT", "1")
from _patches.int8_ext import loader
mod = loader.module(); assert mod is not None; mod.warmup()
dev = "mps"; M, K, N = 4096, 1536, 6144
x8 = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=dev)
wg = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=dev)
wu = torch.randint(-128, 128, (N, K), dtype=torch.int8, device=dev)
rsg = (torch.rand(M, device=dev) * 0.01 + 0.001).float().contiguous()
rsu = (torch.rand(M, device=dev) * 0.01 + 0.001).float().contiguous()

def fused():
    return mod.i8_matmul2d_nt_swiglu(x8, wg, wu, rsg, rsu, None, None, 1)
def unfused():
    g = mod.i8_matmul2d_nt_fused(x8, wg, rsg, None, 0)
    u = mod.i8_matmul2d_nt_fused(x8, wu, rsu, None, 0)
    return torch.nn.functional.silu(g) * u

# Correctness FIRST (review MAJOR #8): never report a speedup for a wrong gate kernel.
# The fused kernel keeps act(gate) in fp32 (single rounding); the correct reference does the
# same — torch silu in fp32 * up in fp32, rounded once (the unfused chain double-rounds the
# silu intermediate to bf16, which is NOT what fusion targets). Tolerance = one bf16 ulp
# (rtol=8e-3) + atol=2e-3 near zero. See docs/superpowers/results/D-results.md.
of = fused()
g = mod.i8_matmul2d_nt_fused(x8, wg, rsg, None, 0)
u = mod.i8_matmul2d_nt_fused(x8, wu, rsu, None, 0)
ref = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16)
torch.mps.synchronize()
md = (of.float() - ref.float()).abs().max().item()
print(f"correctness: max|d|={md:.4g}")
assert torch.allclose(of, ref, atol=2e-3, rtol=8e-3), f"CORRECTNESS FAIL: max|d|={md:.4g}"

def bench(fn, it=50):
    fn(); torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.mps.synchronize(); return (time.perf_counter() - t) / it * 1e3
f, s = bench(fused), bench(unfused)
print(f"M={M} K={K} N={N}: unfused={s:.3f}ms fused={f:.3f}ms speedup={s/f:.2f}x")
if f > s * 1.02:
    print("DECISION (Open Q #4): fused gate is SLOWER -> mark single-pass kernel experimental, "
          "ship the two-fused-calls + activation-multiply fallback as the default path.")
