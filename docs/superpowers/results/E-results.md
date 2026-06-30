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
