# dev/probe_rope_cost.py — anchor the current eager apply_rope cost on a Flux-like shape.
import time, torch
import comfy_kitchen.backends.eager.rope as r
assert torch.backends.mps.is_available()
B,H,L,D = 1,24,4608,128; halfD=D//2
x  = torch.randn(B,H,L,D, device="mps", dtype=torch.bfloat16)
fr = torch.randn(1,1,L,halfD,2,2, device="mps", dtype=torch.float32)   # precomputed table

def bench(fn, it=50, warm=10):
    for _ in range(warm): fn()
    torch.mps.synchronize(); t=time.perf_counter()
    for _ in range(it): fn()
    torch.mps.synchronize(); return (time.perf_counter()-t)/it*1e3

print(f"interleaved apply_rope1     : {bench(lambda: r.apply_rope1(x, fr)):.3f} ms/call")
print(f"split-half apply_rope_split1: {bench(lambda: r.apply_rope_split_half1(x, fr)):.3f} ms/call")
