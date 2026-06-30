# Issue B — empirical / probe results

Environment: M5 Max, macOS 27.0 (arm64), PyTorch 2.11.0, MPS available, Metal 4.1.
Python: `/Volumes/IMPERIAL SPACE/AI/ComfyUI/.venv/bin/python` (cpython 3.12.11).

## Task B.0 — matmul2d operand-dtype probe (HARD GATE)

### B.0 dtype probe (plan's verbatim script, `dev/probe_matmul2d_dtype.py`)

```
matmul2d <half,half,float> : NUMFAIL maxdiff=14.5
matmul2d <bfloat,bfloat,float> : NUMFAIL maxdiff=14.5
matmul2d <float,float,float> : NUMFAIL maxdiff=14.5
```

Diagnosis: the kernel COMPILES and RUNS for all dtypes (no compile/run exception). The
NUMFAIL is a **readout bug in the probe**, not a capability failure. The probe reads the
cooperative-tensor result with a linear `O[i] = c[i]`, but element `i` of a cooperative
tensor maps to a `(row,col)` via `get_multidimensional_index`, not to row-major position
`i`. The linear readout scrambles the result, producing an identical spurious maxdiff
(~14.5, i.e. uncorrelated layout) for EVERY dtype. The production GEMM (Section 2.3)
correctly scatters via `get_multidimensional_index`; the probe did not.

### B.0 dtype probe — CORRECTED readout (`dev/probe_matmul2d_dtype_scatter.py`)

Faithful test using the exact production scatter (`O[row*N+col] = c[i]` with
`row=idx[1], col=idx[0]`), tolerances half<0.5 / bf16<1.0 / fp32<1e-3, plus a non-zero
(SPY) check that the result is not all-zero:

```
matmul2d <half,half,float> (scatter readout) : PASS
matmul2d <bfloat,bfloat,float> (scatter readout) : PASS
matmul2d <float,float,float> (scatter readout) : PASS
```

This matches the handed-down G2 empirical result (all 6 candidates compile+run+CORRECT,
NT layout, float accumulator) exactly. The `<half,half,float>` fp32-accumulate cooperative
destination is CONFIRMED — no design revision (v2 threadgroup-float / .mm setBytes) needed.

**Decision gate result: PASS for half, bfloat, AND float.** `_DT` exposes all three
(`float16`→"half", `bfloat16`→"bfloat", `float32`→"float").

### B.0b dispatch probe (`dev/probe_compile_shader_dispatch.py`)

```
PRM echo: [7, 11, 13, 17, 19] expect [7, 11, 13, 17, 19] -> PASS
grid echo: [0, 100, 200, 1, 101, 201] expect [0, 100, 200, 1, 101, 201] -> PASS
```

`device const int* PRM` scalar passing works under `compile_shader`; the 2D threadgroup
grid maps with no axis transposition (`threads=(gx*TG, gy, 1)` → `tg.x`/`tg.y` as
expected). The GEMM's `threads=(gx*NSG*32, gy, 1), group_size=(NSG*32,1,1)` dispatch is
confirmed correct.
