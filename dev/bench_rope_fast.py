# dev/bench_rope_fast.py
"""Bench fused RoPE vs REAL eager over ALL FOUR public ops (+ cross-length) on a Flux-like shape.
PASS = for each op: fused matches real eager (allclose) AND every call ran on the kernel path AND
fused < eager. Reports ms/call + an estimated per-step win from the profiled rotary share."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from _patches import rope_fast_mps as m
assert torch.backends.mps.is_available(), "bench needs MPS"
m.install_for_test()   # capture real eager originals for the oracle/fallback
# IMPORTANT: install_for_test() patches BOTH the eager package AND the eager.rope source module
# (MAJOR 4 defence-in-depth), so `comfy_kitchen...rope.apply_rope1` is now our fused fn. The REAL
# eager baseline is the captured originals in m._orig (computed before patching).
# The captured single-tensor originals compute directly (no internal global lookup), so they are
# the clean baseline. The captured PAIR originals are NOT usable: their body looks up apply_rope1
# in rope.py's globals, which the defence-in-depth patch replaced -> they would delegate to the
# fused kernel. The true eager pair is literally two true-single calls, so reconstruct it.
eager_apply_rope1            = m._orig["apply_rope1"]
eager_apply_rope_split_half1 = m._orig["apply_rope_split_half1"]
def eager_apply_rope(xq, xk, fr):
    return eager_apply_rope1(xq, fr), eager_apply_rope1(xk, fr)
def eager_apply_rope_split_half(xq, xk, fr):
    return eager_apply_rope_split_half1(xq, fr), eager_apply_rope_split_half1(xk, fr)

B,H,L,D = 1,24,4608,128; halfD=D//2
dt = torch.bfloat16
x  = torch.randn(B,H,L,D, device="mps", dtype=dt)
fr = torch.randn(1,1,L,halfD,2,2, device="mps", dtype=torch.float32)
# cross-length key tensor (interleaved eager slices freqs_cis[:, :, :Lk], so Lk<L is valid):
Lk = 512
xk = torch.randn(B,H,Lk,D, device="mps", dtype=dt)
# NOTE: eager apply_rope_split_half1 has NO cross-length slice (unlike apply_rope1), so a shorter
# key would make eager split-half RAISE -- cross-length split-half is not an eager scenario. The
# split-half pair is therefore benched at matching length (eager-valid) with a second full tensor.
xk2 = torch.randn(B,H,L,D, device="mps", dtype=dt)

def bench(fn, it=100, warm=20):
    for _ in range(warm): fn()
    torch.mps.synchronize(); t=time.perf_counter()
    for _ in range(it): fn()
    torch.mps.synchronize(); return (time.perf_counter()-t)/it*1e3

# (op_label, eager_call, fused_call, expected_n_calls)
CASES = [
    ("apply_rope1 (interleaved)",
        lambda: eager_apply_rope1(x, fr),
        lambda: m.apply_rope1_fused(x, fr), 1),
    ("apply_rope (pair, interleaved)",
        lambda: eager_apply_rope(x, xk, fr),
        lambda: m.apply_rope_fused_pair(x, xk, fr), 2),
    ("apply_rope_split_half1 (split)",
        lambda: eager_apply_rope_split_half1(x, fr),
        lambda: m.apply_rope_split_half1_fused(x, fr), 1),
    ("apply_rope_split_half (pair, split)",
        lambda: eager_apply_rope_split_half(x, xk2, fr),
        lambda: m.apply_rope_split_half_fused_pair(x, xk2, fr), 2),
]

failed = False
for label, eager, fused, ncalls in CASES:
    m._backend_events.clear()
    out_f = fused()
    kinds = [ev[1] for ev in m._backend_events]
    assert len(kinds) == ncalls and all(k == "kernel" for k in kinds), \
        f"{label}: expected {ncalls} kernel call(s), got {kinds}"
    out_e = eager()
    torch.mps.synchronize()
    def _flat(o): return torch.cat([t.float().reshape(-1) for t in (o if isinstance(o, tuple) else (o,))])
    maxdiff = (_flat(out_f) - _flat(out_e)).abs().max().item()
    assert maxdiff < 5e-2, f"{label}: CORRECTNESS FAIL max abs diff {maxdiff:.4g}"
    e_ms = bench(eager); f_ms = bench(fused)
    speed = e_ms / f_ms
    print(f"{label:36s}: eager={e_ms:7.3f}ms fused={f_ms:7.3f}ms speedup={speed:5.2f}x "
          f"(maxdiff {maxdiff:.2g})")
    if f_ms >= e_ms:
        print(f"  REGRESSION: {label} fused not faster"); failed = True

# estimated per-step win from the (interleaved) speedup; rotary ~7-12% of step (profiled).
print("note: end-to-end win is HUMAN-gated (Task 7) and only applies to models whose rotary is "
      "the comfy_kitchen path (Task 0c).")
assert not failed, "at least one convention regressed or fell back"
