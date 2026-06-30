"""Task A.0 — Empirical probes for torch.compile Inductor MPS.

Run ALL probes before starting Tasks A.1–A.4. Record pass/fail inline.

Usage (from repo root):
  TORCH_COMPILE_DEBUG=1 TORCH_LOGS=recompiles \\
    "/Volumes/IMPERIAL SPACE/AI/ComfyUI/.venv/bin/python" \\
    dev/probe_torch_compile_mps.py 2>&1 | tee /tmp/probe_a0.txt

Do NOT set ASFP8_PROFILE=1 or ASFP8_INT8_EXT=1 for probes 0a-0d.
Probes 0e-0f intentionally install patches to test interactions.
"""

import os, sys, subprocess, time, tempfile
import torch
import torch.nn.functional as F

DEVICE = "mps"
DTYPE = torch.float16
B, S, D = 2, 4096, 1536


def make(b=B, s=S, d=D, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(b, s, d, device=DEVICE, dtype=DTYPE, generator=g)
    w = torch.randn(d, device=DEVICE, dtype=DTYPE, generator=g)
    return x, w


def sep():
    print("-" * 60)


# ---------------------------------------------------------------------------
# Probe 0a — F.rms_norm dispatch count (BLOCKER for ADOPT_COMPILE)
# ---------------------------------------------------------------------------

def probe_0a_rms_norm_dispatch():
    """Does Inductor MPS fuse rms_norm+silu+residual into 1 Metal dispatch?

    PASS: bw_tail production shape shows <=1 extern_kernel in lowered IR.
    FAIL: bw_tail shows >=3 extern_kernels (separate dispatches, no true fusion).

    Graph capture (graph_count=1) is necessary but NOT sufficient.
    The extern_kernels count from TORCH_COMPILE_DEBUG=1 is the real test.
    """
    sep()
    print("[Probe 0a] rms_norm+silu+residual dispatch count")
    print("  -> Count 'extern_kernels' in TORCH_COMPILE_DEBUG=1 output.")
    print("     PASS if bw_tail production shape has <=1 extern_kernel.")
    print("     FAIL if >=3 (Inductor emits separate Metal dispatches).")

    import torch._dynamo as dynamo

    def bw_tail(x, w):
        h = F.rms_norm(x, (x.shape[-1],), w, 1e-6)
        h = F.silu(h)
        return x + h

    def rms_only(x, w):
        return F.rms_norm(x, (x.shape[-1],), w, 1e-6)

    for shape_name, (b, s, d) in [
        ("small (1,256,512)", (1, 256, 512)),
        ("production (2,4096,1536)", (B, S, D)),
    ]:
        print(f"\n  Shape: {shape_name}")
        x, w = make(b, s, d)

        # Graph break count (necessary but not sufficient)
        try:
            torch._dynamo.reset()
            expl = dynamo.explain(bw_tail)(x, w)
            print(f"    bw_tail graph_count={expl.graph_count}  "
                  f"breaks={[str(r) for r in expl.break_reasons] or 'none'}")
        except Exception as e:
            print(f"    dynamo.explain failed: {e}")

        # Compile + run bw_tail with fullgraph=True
        try:
            torch._dynamo.reset()
            import torch._inductor.config as ind_cfg
            old_debug = getattr(ind_cfg, "debug", False)
            ind_cfg.debug = True
            try:
                c = torch.compile(bw_tail, backend="inductor", fullgraph=True)
                c(x, w)
                torch.mps.synchronize()
                print(f"    bw_tail compile(fullgraph=True): OK")
            finally:
                ind_cfg.debug = old_debug
        except Exception as e:
            print(f"    bw_tail compile(fullgraph=True): FAILED — "
                  f"{type(e).__name__}: {str(e)[:200]}")

        # rms_only decomposition: does it lower to multiple dispatches on its own?
        try:
            torch._dynamo.reset()
            cr = torch.compile(rms_only, backend="inductor", fullgraph=True)
            cr(x, w)
            torch.mps.synchronize()
            print(f"    rms_only compile(fullgraph=True): OK")
        except Exception as e:
            print(f"    rms_only compile(fullgraph=True): FAILED — "
                  f"{type(e).__name__}: {str(e)[:200]}")


# ---------------------------------------------------------------------------
# Probe 0b — fullgraph=True (BLOCKER)
# ---------------------------------------------------------------------------

def probe_0b_fullgraph():
    """BLOCKER: does fullgraph=True succeed for bw_tail and full_block?

    Default fullgraph=False silently falls back for un-capturable ops.
    fullgraph=True raises immediately on any graph break, making issues visible.
    """
    sep()
    print("[Probe 0b] fullgraph=True compile")

    W = torch.randn(D, D, device=DEVICE, dtype=DTYPE)

    def bw_tail(x, w):
        h = F.rms_norm(x, (x.shape[-1],), w, 1e-6)
        h = F.silu(h)
        return x + h

    def full_block(x, w):
        h = F.rms_norm(x, (x.shape[-1],), w, 1e-6)
        h = (h.reshape(-1, D) @ W.T).reshape(x.shape)
        h = F.silu(h)
        return x + h

    x, w = make()
    for name, fn in [("bw_tail", bw_tail), ("full_block", full_block)]:
        torch._dynamo.reset()
        try:
            c = torch.compile(fn, backend="inductor", fullgraph=True)
            out = c(x, w)
            torch.mps.synchronize()
            print(f"  {name} fullgraph=True: PASS")
        except Exception as e:
            print(f"  {name} fullgraph=True: FAIL — {type(e).__name__}: {str(e)[:200]}")


# ---------------------------------------------------------------------------
# Probe 0c — dynamic=True shapes (BLOCKER if follow-on wraps real DiT modules)
# ---------------------------------------------------------------------------

def probe_0c_dynamic_shapes():
    """BLOCKER for auto-wiring: does dynamic=True recompile per sequence length?

    Run with TORCH_LOGS=recompiles to see recompilation events.
    PASS: no 'Recompiling' in output when switching between shapes.
    FAIL (use-per-fixed-shape only): recompile happens at each new S.
    """
    sep()
    print("[Probe 0c] dynamic=True cross-shape recompile check")
    print("  (set TORCH_LOGS=recompiles in env to see recompile events)")

    def bw_tail(x, w):
        h = F.rms_norm(x, (x.shape[-1],), w, 1e-6)
        h = F.silu(h)
        return x + h

    torch._dynamo.reset()
    try:
        c = torch.compile(bw_tail, backend="inductor", dynamic=True)
        w = torch.randn(D, device=DEVICE, dtype=DTYPE)

        x1, _ = make(B, S, D)
        c(x1, w)
        torch.mps.synchronize()
        print(f"  Shape (2,4096,1536): OK")

        x2, _ = make(B, 9216, D)
        c(x2, w)
        torch.mps.synchronize()
        print(f"  Shape (2,9216,1536): OK")

        x3, _ = make(B, S, D)
        c(x3, w)
        torch.mps.synchronize()
        print(f"  Shape (2,4096,1536) again: OK")

        print("  Check TORCH_LOGS=recompiles output for 'Recompiling' entries.")
        print("  PASS: zero recompiles on shape (2,4096,1536) after (2,9216,1536).")
    except Exception as e:
        print(f"  dynamic=True FAILED: {type(e).__name__}: {str(e)[:300]}")


# ---------------------------------------------------------------------------
# Probe 0d — subprocess cold compile latency (BLOCKER for ComfyUI wiring)
# ---------------------------------------------------------------------------

def probe_0d_cold_latency():
    """BLOCKER for ComfyUI wiring: real cold compile latency from a fresh process.

    In-process torch._dynamo.reset() does not simulate a cold cache state.
    This probe spawns a subprocess so the OS starts fresh, and uses a
    temporary TORCHINDUCTOR_CACHE_DIR to isolate cold vs warm persistent cache.

    PASS: cold < 30s (acceptable for model load time).
    FAIL: cold > 60s (unacceptable for ComfyUI startup).
    """
    sep()
    print("[Probe 0d] Cold compile latency (subprocess, fresh cache)")

    script = """
import os, time, torch, torch.nn.functional as F
DEVICE="mps"; DTYPE=torch.float16; B,S,D=2,4096,1536
# Allocate inputs BEFORE starting timer (timer measures compile only)
x = torch.randn(B,S,D,device=DEVICE,dtype=DTYPE)
w = torch.randn(D,device=DEVICE,dtype=DTYPE)
def bw_tail(x,w):
    h=F.rms_norm(x,(x.shape[-1],),w,1e-6)
    h=F.silu(h)
    return x+h
c=torch.compile(bw_tail,backend="inductor")
torch.mps.synchronize()
t0=time.perf_counter()
c(x,w)
torch.mps.synchronize()
t1=time.perf_counter()
print(f"cold_ms={(t1-t0)*1e3:.0f}")
"""
    python = "/Volumes/IMPERIAL SPACE/AI/ComfyUI/.venv/bin/python"

    # Cold run (empty cache)
    with tempfile.TemporaryDirectory() as cache1:
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = cache1
        r = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, env=env, timeout=300,
        )
        cold_lines = [l for l in r.stdout.splitlines() if "cold_ms" in l]
        if cold_lines:
            print(f"  Cold (empty cache): {cold_lines[-1]}")
        else:
            print(f"  Cold run failed. stderr: {r.stderr[-500:] if r.stderr else 'none'}")

    # Warm persistent cache (run twice, second = cache hit)
    with tempfile.TemporaryDirectory() as cache2:
        env2 = os.environ.copy()
        env2["TORCHINDUCTOR_CACHE_DIR"] = cache2
        subprocess.run([python, "-c", script], capture_output=True, text=True,
                       env=env2, timeout=300)
        r2 = subprocess.run([python, "-c", script], capture_output=True, text=True,
                            env=env2, timeout=300)
        warm_lines = [l for l in r2.stdout.splitlines() if "cold_ms" in l]
        if warm_lines:
            print(f"  Warm (persistent cache hit): {warm_lines[-1]}")
        else:
            print(f"  Warm run failed. stderr: {r2.stderr[-500:] if r2.stderr else 'none'}")

    print("  PASS if cold_ms < 30000. FAIL if cold_ms > 60000.")


# ---------------------------------------------------------------------------
# Probe 0e — rmsnorm_mps_large.install() interaction (BLOCKER for ADOPT_COMPILE)
# ---------------------------------------------------------------------------

def probe_0e_rmsnorm_patch_interaction():
    """BLOCKER: does Dynamo graph-break when rmsnorm_mps_large is installed?

    rmsnorm_mps_large globally replaces F.rms_norm with a Python wrapper
    containing control-flow (device check + row-count threshold). Dynamo may
    or may not be able to trace through this, depending on the Python ops used.

    PASS: graph_count=1 on both fast branch and forced manual branch.
    FAIL: graph_count > 1 or exception — ADOPT_COMPILE scoped to unpatched only.
    """
    sep()
    print("[Probe 0e] rmsnorm_mps_large.install() + compile(bw_tail)")

    sys.path.insert(0, "/Volumes/IMPERIAL SPACE/AI/ComfyUI/custom_nodes/ComfyUI-AppleSilicon-FP8")

    try:
        import _patches.rmsnorm_mps_large as rml
        rml.install()
        print(f"  rmsnorm_mps_large installed (_THRESHOLD=2^21={rml._THRESHOLD})")
    except Exception as e:
        print(f"  rmsnorm_mps_large install FAILED: {e} — probe skipped")
        return

    import torch._dynamo as dynamo

    def bw_tail(x, w):
        h = F.rms_norm(x, (x.shape[-1],), w, 1e-6)
        h = F.silu(h)
        return x + h

    # Fast branch (rows << _THRESHOLD, uses stock fused kernel)
    x_small, w_small = make(1, 64, 64)
    torch._dynamo.reset()
    try:
        expl = dynamo.explain(bw_tail)(x_small, w_small)
        print(f"  Fast branch (rows=64): graph_count={expl.graph_count}  "
              f"breaks={[str(r) for r in expl.break_reasons] or 'none'}")
    except Exception as e:
        print(f"  Fast branch explain FAILED: {e}")

    # Forced manual branch (set _THRESHOLD=0 so all inputs hit fp32 path)
    try:
        orig_thresh = rml._THRESHOLD
        rml._THRESHOLD = 0
        torch._dynamo.reset()
        try:
            expl2 = dynamo.explain(bw_tail)(x_small, w_small)
            print(f"  Manual branch (thresh=0): graph_count={expl2.graph_count}  "
                  f"breaks={[str(r) for r in expl2.break_reasons] or 'none'}")
        finally:
            rml._THRESHOLD = orig_thresh
    except AttributeError:
        print("  _THRESHOLD attribute not found on rml module — inspect patch internals")
    except Exception as e:
        print(f"  Manual branch test FAILED: {e}")

    print("  PASS if graph_count=1 on both branches.")
    print("  FAIL if graph_count>1 — ADOPT_COMPILE limited to unpatched synthetic functions.")


# ---------------------------------------------------------------------------
# Probe 0f — ASFP8_PROFILE=1 graph break (MAJOR operational incompatibility)
# ---------------------------------------------------------------------------

def probe_0f_profile_graph_break():
    """Do mps_profile wrappers (F.rms_norm, F.linear, torch.matmul, etc.) cause graph breaks?

    mps_profile wraps: F.rms_norm, F.linear, F.conv2d, F.conv3d, F.layer_norm,
                       torch.matmul, torch.bmm.
    It does NOT wrap F.silu.

    PASS: graph_count=1 (wrappers are transparent to Dynamo).
    FAIL: graph_count>1 — ASFP8_PROFILE=1 and ASFP8_TORCH_COMPILE=1 mutually exclusive.
    """
    sep()
    print("[Probe 0f] ASFP8_PROFILE=1 graph break check")

    sys.path.insert(0, "/Volumes/IMPERIAL SPACE/AI/ComfyUI/custom_nodes/ComfyUI-AppleSilicon-FP8")
    os.environ["ASFP8_PROFILE"] = "1"

    try:
        import _patches.mps_profile as prof
        prof.install()
        print("  mps_profile installed (wraps F.rms_norm, F.linear, "
              "torch.matmul, torch.bmm, SDPA, layernorm, convs)")
    except Exception as e:
        print(f"  mps_profile install FAILED: {e} — probe skipped")
        os.environ.pop("ASFP8_PROFILE", None)
        return

    import torch._dynamo as dynamo

    def bw_tail(x, w):
        h = F.rms_norm(x, (x.shape[-1],), w, 1e-6)
        h = F.silu(h)
        return x + h

    x, w = make(1, 256, 512)
    torch._dynamo.reset()
    try:
        expl = dynamo.explain(bw_tail)(x, w)
        print(f"  With mps_profile: graph_count={expl.graph_count}  "
              f"breaks={[str(r) for r in expl.break_reasons] or 'none'}")
        if expl.graph_count > 1:
            print("  -> ASFP8_PROFILE=1 and ASFP8_TORCH_COMPILE=1 are MUTUALLY EXCLUSIVE.")
        else:
            print("  -> mps_profile wrappers are transparent to Dynamo. Compatible.")
    except Exception as e:
        print(f"  explain with mps_profile FAILED: {e}")
    finally:
        os.environ.pop("ASFP8_PROFILE", None)


# ---------------------------------------------------------------------------
# Probe 0g — fp16 rms_norm fp32 upcast check (MAJOR correctness gate)
# ---------------------------------------------------------------------------

def probe_0g_fp16_rmsnorm_precision():
    """Does Inductor upcast rms_norm reduction to fp32, or run it in fp16?

    Absence of NaN on random data is not sufficient. This probe uses stress
    inputs with large magnitude (amplifies fp16 vs fp32 difference) and compares
    compiled output against a manual fp32 reference.

    PASS:  max_abs < 0.1 (compiled output close to fp32 reference; fp32 upcast likely).
    WARN:  0.1 <= max_abs < 1.0 (fp16 accumulation suspected; borderline).
    FAIL:  max_abs >= 1.0 (fp16 accumulation confirmed; correctness risk on large magnitudes).
    """
    sep()
    print("[Probe 0g] fp16 rms_norm fp32 upcast check (stress inputs, fp32 reference)")

    def rms_only(x, w):
        return F.rms_norm(x, (x.shape[-1],), w, 1e-6)

    # Stress inputs: large magnitude to amplify fp16 vs fp32 difference
    # Use CPU fp32 for ground truth, then cast to fp16 for MPS
    x_cpu = torch.randn(2, 4096, 1536, dtype=torch.float32) * 10.0
    x_mps = x_cpu.half().to(DEVICE)
    w_cpu = torch.ones(1536, dtype=torch.float32)
    w_mps = w_cpu.half().to(DEVICE)

    # fp32 CPU reference (ground truth)
    with torch.no_grad():
        ref_fp32 = rms_only(x_cpu, w_cpu)

    # eager fp16 on MPS for comparison baseline
    with torch.no_grad():
        out_eager = rms_only(x_mps, w_mps)
    torch.mps.synchronize()
    eager_diff = (out_eager.float().cpu() - ref_fp32).abs().max().item()
    print(f"  Eager fp16 vs fp32 ref: max_abs={eager_diff:.4f}")

    # compiled fp16 on MPS
    torch._dynamo.reset()
    try:
        c = torch.compile(rms_only, backend="inductor")
        c(x_mps, w_mps)  # warmup/compile
        torch.mps.synchronize()
        with torch.no_grad():
            out_compiled = c(x_mps, w_mps)
        torch.mps.synchronize()
        compiled_diff = (out_compiled.float().cpu() - ref_fp32).abs().max().item()
        print(f"  Compiled fp16 vs fp32 ref: max_abs={compiled_diff:.4f}")

        if compiled_diff < 0.1:
            print("  PASS — compiled output close to fp32 ref (fp32 accumulation likely).")
        elif compiled_diff < 1.0:
            print("  WARN — moderate diff; fp16 accumulation suspected but borderline.")
        else:
            print("  FAIL — large diff; fp16 accumulation confirmed; correctness risk.")
    except Exception as e:
        print(f"  compile FAILED: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Task A.0 — Empirical probes for torch.compile Inductor MPS")
    print(f"PyTorch {torch.__version__} | MPS: {torch.backends.mps.is_available()}")
    print("=" * 60)

    if not torch.backends.mps.is_available():
        print("MPS not available. Cannot run probes.")
        sys.exit(1)

    probe_0a_rms_norm_dispatch()
    probe_0b_fullgraph()
    probe_0c_dynamic_shapes()
    probe_0d_cold_latency()
    probe_0e_rmsnorm_patch_interaction()
    probe_0f_profile_graph_break()
    probe_0g_fp16_rmsnorm_precision()

    sep()
    print("\nAll probes complete.")
    print("Record results in docs/superpowers/results/A-results.md before proceeding to Task A.1.")
    print("Gate: ADOPT_COMPILE requires Probe 0a dispatch_count<=1 AND Probe 0b OK.")


if __name__ == "__main__":
    main()
