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
