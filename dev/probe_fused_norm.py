# dev/probe_fused_norm.py
"""Task 0 empirical probes for issue E. Compiles + runs the three compile_shader facts the
fused-norm kernel depends on, prints PASS/FAIL for each, and exits non-zero if a BLOCKER probe
fails. Run BEFORE implementing the kernel; record the output in the commit message."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

if not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")):
    print("SKIP: needs MPS + torch.mps.compile_shader")
    sys.exit(0)

failures = []

# ---- probe #1: grid.y ~= 2^24 dispatch (does a giant non-x grid dimension work?) ----
GRID_SRC = r"""
#include <metal_stdlib>
using namespace metal;
kernel void grid_probe(device uint* out [[buffer(0)]],
                       uint3 tgid [[threadgroup_position_in_grid]]) {
    out[tgid.y] = tgid.y;
}
"""
try:
    rows = 1 << 24
    out = torch.zeros(rows, dtype=torch.int32, device="mps")  # ~64 MiB
    lib = torch.mps.compile_shader(GRID_SRC)
    lib.grid_probe(out, threads=(1, rows, 1), group_size=(1, 1, 1))
    torch.mps.synchronize()
    idx = [0, rows // 2, rows - 1]
    ok = all(int(out[i].item()) == i for i in idx)
    print(f"PROBE#1 grid.y=2^24 single-dim dispatch: {'PASS' if ok else 'FAIL'} "
          f"(checked {idx} -> {[int(out[i].item()) for i in idx]})")
    if not ok:
        print("  -> NOTE: production kernel z-tiles anyway (row = z*ny + y), so this is informational.")
except Exception as e:
    print(f"PROBE#1 grid.y=2^24 single-dim dispatch: FAIL/raised ({e})")
    print("  -> NOTE: production kernel z-tiles anyway; informational, not a blocker.")

# ---- probe #2: Python scalar -> constant int&/float& binding (optional simplification) ----
SCALAR_SRC = r"""
#include <metal_stdlib>
using namespace metal;
kernel void scalar_probe(device float* out [[buffer(0)]],
                         constant int& n [[buffer(1)]],
                         constant float& eps [[buffer(2)]]) {
    out[0] = float(n) + eps;
}
"""
try:
    out = torch.zeros(1, dtype=torch.float32, device="mps")
    lib = torch.mps.compile_shader(SCALAR_SRC)
    lib.scalar_probe(out, 7, 0.5, threads=(1, 1, 1), group_size=(1, 1, 1))
    torch.mps.synchronize()
    ok = abs(float(out[0].item()) - 7.5) < 1e-4
    print(f"PROBE#2 constant& scalar binding: {'PASS (simplification available)' if ok else 'FAIL'}")
    print("  -> design uses device-tensor meta/epsb regardless; this is informational only.")
except Exception as e:
    print(f"PROBE#2 constant& scalar binding: NOT AVAILABLE ({e}) -> confirms device-buffer design")

# ---- probe #3: bfloat literal cast in MSL (BLOCKER for treating bf16 tests as success) ----
BF16_SRC = r"""
#include <metal_stdlib>
using namespace metal;
kernel void bf16_cast(device bfloat* out [[buffer(0)]]) {
    out[0] = bfloat(1.25f);
}
"""
try:
    out = torch.zeros(1, dtype=torch.bfloat16, device="mps")
    lib = torch.mps.compile_shader(BF16_SRC)
    lib.bf16_cast(out, threads=(1, 1, 1), group_size=(1, 1, 1))
    torch.mps.synchronize()
    ok = abs(float(out[0].float().item()) - 1.25) < 1e-2
    print(f"PROBE#3 bfloat(y) cast compiles+runs: {'PASS' if ok else 'FAIL'} (got {float(out[0].float().item())})")
    if not ok:
        failures.append("probe#3 bfloat cast")
except Exception as e:
    print(f"PROBE#3 bfloat(y) cast compiles+runs: FAIL ({e})")
    failures.append("probe#3 bfloat cast")

if failures:
    print(f"BLOCKER probe(s) failed: {failures} -> bf16 path must be disabled or fixed before use.")
    sys.exit(1)
print("Task 0 probes done.")
