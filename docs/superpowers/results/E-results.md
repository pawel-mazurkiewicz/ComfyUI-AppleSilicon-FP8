# Issue E — Fused RMSNorm+modulation+residual kernel — empirical results

Environment: M5 Max, macOS 27, PyTorch 2.11.0, MPS available, `torch.mps.compile_shader` present.
Python: `/Volumes/IMPERIAL SPACE/AI/ComfyUI/.venv/bin/python`

## Task 0 — compile_shader probes (`dev/probe_fused_norm.py`)

```
PROBE#1 grid.y=2^24 single-dim dispatch: PASS (checked [0, 8388608, 16777215] -> [0, 8388608, 16777215])
PROBE#2 constant& scalar binding: PASS (simplification available)
  -> design uses device-tensor meta/epsb regardless; this is informational only.
PROBE#3 bfloat(y) cast compiles+runs: PASS (got 1.25)
Task 0 probes done.
```

- PROBE#1 PASS: a single grid.y dimension of 2^24 dispatches correctly (z-tiling still used in production by design).
- PROBE#2 PASS: scalar `constant&` binding works (informational; device-buffer design kept).
- PROBE#3 PASS: `bfloat(literal)` compiles+runs => bf16 parametrization enabled in Task E.1.

## Task E.1/E.2 — correctness (kernel path proven via `_last_backend == "kernel"`)

- 14 non-slow tests PASS (fp16/bf16/fp32 full path, tight bare-norm, optional-arg combos,
  per-batch + mixed-group adaLN, multidim `F.rms_norm` reroute, forced-fallback group-equiv,
  bad-optional-shape -> fallback, cpu fallback).
- `test_stock_mps_broken_regime_2pow22` PASS (1<<22 rows, kernel path).
- `test_overflow_rows_2pow24_kernel_path` PASS — rows*D = 1<<24 * 256 = 4.29e9 > 2**31;
  `_reference` monkeypatched to raise so it CANNOT pass through the fallback => proves the
  real 64-bit-indexed kernel handled the int32-offset overflow regime.
- ruff check: All checks passed.

Deviation from plan: `test_bad_optional_shape_falls_back` — the plan's verbatim body called the
wrapper with a genuinely-malformed weight/residual then asserted `_last_backend == "fallback"`,
but the torch fallback legitimately broadcast-raises on the invalid tensor before the assert runs.
Verified empirically the kernel is NOT dispatched (flag set to "fallback" before the raise), so the
test now wraps each malformed call in `pytest.raises(RuntimeError)` and asserts
`_last_backend == "fallback"`. No kernel correctness contract loosened.

## Task E.3 — benchmark (`dev/bench_fused_norm.py`, rows=1<<20, dim=1536, fp16)

```
correctness OK (max abs diff 0.0625, backend=kernel)
first-call (compile_shader) latency: 517.3 ms
rows=1048576 dim=1536 dtype=torch.float16
separate=198.572ms  fused=20.819ms  speedup=9.54x  fused~464 GB/s (BW estimate only; no measured roofline)
```

Correctness + `_last_backend == "kernel"` asserted BEFORE timing. `fused < separate` holds (9.54x).

TG tuning (Step 4) skipped: TG=128 kept; 9.54x already well above the ~2-4x target.

## Task E.4 — registration (opt-in, never fatal)

- `noop: False` with `ASFP8_FUSED_NORM` unset (install is a no-op).
- `active: True rerouted: True` with `ASFP8_FUSED_NORM=1` (F.rms_norm rerouted to `_rms_norm`).
- `bisection-name resolvable: fused_norm_mps` (matches `ASFP8_ENABLE_ONLY`/`ASFP8_DISABLE` filter).
- Top-level package import + install: the package-internal `_patches.fused_norm_mps._installed`
  is True and `F.rms_norm` is rerouted to the package module's `_rms_norm` (the active log line
  prints). NOTE: the plan's verbatim Step-6 one-liner printed `installed via package import: False`
  only because this worktree's directory name is the random `wf_*` slug (not
  `ComfyUI-AppleSilicon-FP8`), which causes `from _patches import ...` to load a SECOND, distinct
  copy of the module. Inspecting `sys.modules['<pkg>._patches.fused_norm_mps']` (the copy the
  package actually installed) shows `_installed: True`. Wiring is correct.
- Non-slow suite re-run after registration: 14 passed, 0 failed.
