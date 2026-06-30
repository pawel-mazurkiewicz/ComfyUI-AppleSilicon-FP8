"""Issue A benchmark: Inductor MPS fusion speedup on the DiT-block bandwidth-bound tail.

Measures three configurations:
  1. eager  — stock torch ops, no compile (baseline)
  2. compiled (bw-tail only) — rms_norm + silu + residual, no linear
  3. compiled (full block)   — rms_norm + linear + silu + residual

Correctness is asserted BEFORE the timing loop. DECISION: CORRECTNESS_FAIL
is emitted if compiled output diverges, regardless of speed.

ADOPT_COMPILE requires BOTH:
  - bw_tail speedup >= 1.5x (timing gate)
  - Probe 0a dispatch_count <= 1 (fusion gate, confirmed in Task A.0)

Run (from repo root, no ASFP8_PROFILE=1):
  /Volumes/IMPERIAL\\ SPACE/AI/ComfyUI/.venv/bin/python dev/bench_torch_compile_fusion.py

NOTE: First run includes Inductor compile time. Subsequent runs use the kernel
cache. Warmup is done inside each bench() call to ensure only steady-state
execution is timed.
"""

import time
import sys

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE = "mps"
DTYPE = torch.float16
B, S, D = 2, 4096, 1536   # production DiT shape
WARMUP = 5
ITERS = 50

DECISION_THRESHOLD = 1.5   # bw-tail speedup threshold to adopt compile


# ---------------------------------------------------------------------------
# Blocks under test
# ---------------------------------------------------------------------------

def bw_tail(x, weight, W):
    """Bandwidth-bound tail: rms_norm + silu + residual. No linear."""
    h = F.rms_norm(x, (x.shape[-1],), weight, 1e-6)
    h = F.silu(h)
    return x + h


def full_block(x, weight, W):
    """Full DiT block tail: rms_norm + linear + silu + residual."""
    h = F.rms_norm(x, (x.shape[-1],), weight, 1e-6)
    h = (h.reshape(-1, h.shape[-1]) @ W.T).reshape(x.shape)
    h = F.silu(h)
    return x + h


# ---------------------------------------------------------------------------
# Fusion inspection
# ---------------------------------------------------------------------------

def inspect_graphs(fn, x, weight, W):
    """Return (graph_count, break_reasons) for fn via torch._dynamo.explain."""
    import torch._dynamo as dynamo
    try:
        explanation = dynamo.explain(fn)(x, weight, W)
        return explanation.graph_count, [str(r) for r in explanation.break_reasons]
    except Exception as e:
        return -1, [str(e)]


# ---------------------------------------------------------------------------
# Correctness assertion (run BEFORE timing loop)
# ---------------------------------------------------------------------------

def assert_correctness(compiled_fn, eager_fn, x, weight, W, name):
    """Assert compiled output matches eager within fp16 tolerance.

    Returns True on pass, False on fail (never raises — caller records result).
    """
    try:
        eager_out = eager_fn(x, weight, W)
        compiled_out = compiled_fn(x, weight, W)
        torch.mps.synchronize()
        torch.testing.assert_close(
            compiled_out, eager_out, atol=1e-2, rtol=1e-2,
        )
        print(f"  [{name}] correctness: PASS "
              f"(max_abs={(compiled_out - eager_out).abs().max().item():.4f})")
        return True
    except AssertionError as e:
        print(f"  [{name}] correctness: FAIL — {e}")
        return False
    except Exception as e:
        print(f"  [{name}] correctness: ERROR — {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Bench harness — mirrors dev/bench_gemm.py
# ---------------------------------------------------------------------------

def bench(fn, x, weight, W, warmup=WARMUP, iters=ITERS):
    """Return ms/call (steady-state, after warmup). Uses true GPU time via sync."""
    for _ in range(warmup):
        fn(x, weight, W)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x, weight, W)
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def measure_compile_latency(fn, make_inputs_fn):
    """Measure first-call (cold) compile+execute latency in ms.

    Inputs are allocated BEFORE the timer starts so the measurement
    captures compile + first-dispatch only (not tensor allocation).
    """
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="inductor")
    x, weight, W = make_inputs_fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    compiled(x, weight, W)
    torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1e3


def make_inputs(seed=42):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE, generator=g)
    weight = torch.randn(D, device=DEVICE, dtype=DTYPE, generator=g)
    W = torch.randn(D, D, device=DEVICE, dtype=DTYPE, generator=g)
    return x, weight, W


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sep(width=70): print("-" * width)

def main():
    if not torch.backends.mps.is_available():
        print("MPS not available — cannot benchmark. Exiting.")
        sys.exit(1)

    print(f"Issue A — torch.compile Inductor MPS fusion benchmark")
    print(f"PyTorch {torch.__version__} | device={DEVICE} | dtype={DTYPE} | shape=({B},{S},{D})")
    print(f"warmup={WARMUP} iters={ITERS} | decision threshold={DECISION_THRESHOLD}x")
    sep()

    x, weight, W = make_inputs()

    # ------------------------------------------------------------------
    # 1. Fusion inspection
    # ------------------------------------------------------------------
    print("\n[1] Fusion inspection (torch._dynamo.explain)")
    print("    Note: graph_count=1 is necessary but NOT sufficient for fusion.")
    print("    True fusion requires dispatch_count<=1 in Probe 0a (Task A.0).")
    for name, fn in [("bw_tail", bw_tail), ("full_block", full_block)]:
        gc, reasons = inspect_graphs(fn, x, weight, W)
        print(f"  {name:12s}: graph_count={gc}  breaks={reasons or 'none'}")

    # ------------------------------------------------------------------
    # 2. Compile latency (cold first call, inputs pre-allocated)
    # ------------------------------------------------------------------
    print("\n[2] First-call (cold) compile latency")
    try:
        cold_bw = measure_compile_latency(bw_tail, make_inputs)
        print(f"  bw_tail   cold compile+run: {cold_bw:.0f} ms")
    except Exception as e:
        print(f"  bw_tail   COMPILE FAILED: {e}")
        cold_bw = None

    torch._dynamo.reset()
    try:
        cold_full = measure_compile_latency(full_block, make_inputs)
        print(f"  full_block cold compile+run: {cold_full:.0f} ms")
    except Exception as e:
        print(f"  full_block COMPILE FAILED: {e}")
        cold_full = None

    # ------------------------------------------------------------------
    # 3. Steady-state benchmark (correctness first, then timing)
    # ------------------------------------------------------------------
    print("\n[3] Steady-state benchmark (correctness asserted before timing)")
    sep()

    results = {}
    correctness_ok = {}

    # -- bw_tail eager --
    eager_bw_ms = bench(bw_tail, x, weight, W)
    results["eager_bw_ms"] = eager_bw_ms
    print(f"  bw_tail   eager:    {eager_bw_ms:.3f} ms")

    # -- bw_tail compiled --
    try:
        torch._dynamo.reset()
        compiled_bw = torch.compile(bw_tail, backend="inductor")
        # warmup (triggers compilation on first call)
        for _ in range(WARMUP):
            compiled_bw(x, weight, W)
        torch.mps.synchronize()

        # Correctness BEFORE timing
        correctness_ok["bw_tail"] = assert_correctness(
            compiled_bw, bw_tail, x, weight, W, "bw_tail"
        )

        comp_bw_ms = bench(compiled_bw, x, weight, W, warmup=0)
        results["comp_bw_ms"] = comp_bw_ms
        speedup_bw = eager_bw_ms / comp_bw_ms
        results["speedup_bw"] = speedup_bw
        print(f"  bw_tail   compiled: {comp_bw_ms:.3f} ms  speedup={speedup_bw:.2f}x")
    except Exception as e:
        print(f"  bw_tail   compiled: FAILED ({type(e).__name__}: {str(e)[:100]})")
        results["comp_bw_ms"] = None
        results["speedup_bw"] = None
        correctness_ok["bw_tail"] = False

    # -- full_block eager --
    eager_full_ms = bench(full_block, x, weight, W)
    results["eager_full_ms"] = eager_full_ms
    print(f"  full_block eager:    {eager_full_ms:.3f} ms")

    # -- full_block compiled --
    try:
        torch._dynamo.reset()
        compiled_full = torch.compile(full_block, backend="inductor")
        for _ in range(WARMUP):
            compiled_full(x, weight, W)
        torch.mps.synchronize()

        # Correctness BEFORE timing
        correctness_ok["full_block"] = assert_correctness(
            compiled_full, full_block, x, weight, W, "full_block"
        )

        comp_full_ms = bench(compiled_full, x, weight, W, warmup=0)
        results["comp_full_ms"] = comp_full_ms
        speedup_full = eager_full_ms / comp_full_ms
        results["speedup_full"] = speedup_full
        print(f"  full_block compiled: {comp_full_ms:.3f} ms  speedup={speedup_full:.2f}x")
    except Exception as e:
        print(f"  full_block compiled: FAILED ({type(e).__name__}: {str(e)[:100]})")
        results["comp_full_ms"] = None
        results["speedup_full"] = None
        correctness_ok["full_block"] = False

    # ------------------------------------------------------------------
    # 4. Roofline check (presented as range, not a single confident bound)
    # ------------------------------------------------------------------
    print("\n[4] Bandwidth roofline analysis (range — see Task A.0 Probe 0a for dispatch truth)")
    # rms_norm bytes: optimistic (1 read) vs conservative (2 reads)
    per_tensor = B * S * D * 2  # bytes for one fp16 tensor
    # Optimistic: rms_norm = 1 read + 1 write; silu = 1+1; residual = 2+1
    eager_bytes_optimistic = per_tensor * (2 + 2 + 3)
    # Conservative: rms_norm = 2 reads + 1 write; silu = 1+1; residual = 2+1
    eager_bytes_conservative = per_tensor * (3 + 2 + 3)
    fused_bytes = per_tensor * 3   # read x + read res + write out
    bw_gbps = 400.0
    ro_eager_opt = eager_bytes_optimistic / (bw_gbps * 1e9) * 1e3
    ro_eager_con = eager_bytes_conservative / (bw_gbps * 1e9) * 1e3
    ro_fused = fused_bytes / (bw_gbps * 1e9) * 1e3
    print(f"  Roofline (M5 Max @400 GB/s):")
    print(f"    eager  3-kernel min (optimistic):     {ro_eager_opt:.3f} ms  ({eager_bytes_optimistic/1e6:.0f} MB)")
    print(f"    eager  3-kernel min (conservative):   {ro_eager_con:.3f} ms  ({eager_bytes_conservative/1e6:.0f} MB)")
    print(f"    fused  1-kernel min:                  {ro_fused:.3f} ms  ({fused_bytes/1e6:.0f} MB)")
    print(f"    max achievable speedup range: {ro_eager_opt/ro_fused:.2f}x – {ro_eager_con/ro_fused:.2f}x")
    if results.get("comp_bw_ms") and results.get("eager_bw_ms"):
        achieved = results["eager_bw_ms"] / results["comp_bw_ms"]
        print(f"    achieved: {achieved:.2f}x vs theoretical range "
              f"{ro_eager_opt/ro_fused:.2f}–{ro_eager_con/ro_fused:.2f}x")

    # ------------------------------------------------------------------
    # 5. Decision (requires BOTH speedup AND probe-0a dispatch_count)
    # ------------------------------------------------------------------
    print("\n[5] Decision")
    sep()
    speedup_bw = results.get("speedup_bw")
    bw_correctness = correctness_ok.get("bw_tail", False)

    if not bw_correctness:
        decision = "CORRECTNESS_FAIL"
        detail = (
            "bw_tail compiled output diverged from eager. "
            "Do NOT adopt compile. Issues D and E proceed at FULL scope."
        )
    elif speedup_bw is None:
        decision = "COMPILE_FAILED"
        detail = "Inductor MPS raised an error. Issues D and E proceed at FULL scope."
    elif speedup_bw >= DECISION_THRESHOLD:
        decision = "ADOPT_COMPILE_PENDING_PROBE_0A"
        detail = (
            f"bw-tail speedup={speedup_bw:.2f}x >= {DECISION_THRESHOLD}x threshold "
            f"AND correctness passes. "
            "FINAL DECISION requires Probe 0a dispatch_count <= 1 (check Task A.0 results). "
            "If dispatch_count > 1, speedup reflects Python-overhead reduction only "
            "(not DRAM traffic reduction) -> downgrade to HAND_KERNELS_NEEDED. "
            "If dispatch_count <= 1 -> ADOPT_COMPILE: Issues D and E SHRINK to "
            "'only what compile leaves behind' (primarily GEMM epilogue fusions)."
        )
    else:
        decision = "HAND_KERNELS_NEEDED"
        detail = (
            f"bw-tail speedup={speedup_bw:.2f}x < {DECISION_THRESHOLD}x threshold. "
            "Inductor MPS generates separate Metal kernels rather than a fused one. "
            "Issues D and E proceed at FULL scope."
        )

    print(f"  DECISION: {decision}")
    print(f"  {detail}")
    sep()
    print("\nPaste this output into the commit message for issue-A.")
    print("Cross-reference with Task A.0 Probe 0a dispatch count before finalising.")


if __name__ == "__main__":
    main()
