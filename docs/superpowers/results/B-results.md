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

## Task B.1 — baseline stock conv vs measured same-shape GEMM (`dev/bench_mps_conv.py`)

Device: Apple M5 Max, 18 cores (6 Super + 12 Performance), macOS 27.

```
conv2d 3x3 256ch 512x512: 5.115 ms  60458.8 GFLOP/s
  measured GEMM @ M=262144,K=2304,N=256: 5.683 ms  54417.2 GFLOP/s (conv2d achieves 111% of achieved GEMM)
conv3d 3x3x3 128ch 5x256x256: 69.926 ms  4146.0 GFLOP/s
  measured GEMM @ M=327680,K=3456,N=128: 5.456 ms  53139.6 GFLOP/s (conv3d achieves 8% of achieved GEMM)
driver_allocated (current, NOT peak): 4.32 GiB
```

DECISION (data-driven, Open Question #8): stock conv2d already achieves 111% of the
achieved same-shape GEMM -> conv2d is ALREADY competitive; do NOT route conv2d. Stock
conv3d achieves only 8% of the same-shape GEMM (~13x headroom) -> conv3d is the firm win.
**Scope install to conv3d-only (`ASFP8_CONV_IM2COL=3d`).** conv2d path is still built and
correctness-tested (B.4) but not installed by default.

## Task B.5 — conv2d verify-first bench + deterministic tile-cap test

conv2d bench (M5 Max, fp16, 1x256x512x512, w=256x256x3x3, pad=1):
```
  correctness: max|diff|=0.123 -> OK
conv2d  im2col=24.118ms  stock=5.066ms  speedup=0.21x
  current_allocated (NOT peak): 0.63GiB
```
Correctness OK (gate passes before timing). Speed 0.21x = ~4.7x SLOWER than stock conv2d,
exactly as B.1 predicted (stock conv2d already at 111% of the achieved same-shape GEMM).
DOCUMENTED non-ship outcome: conv2d is NOT installed by default (install gated to 3d-only).
No regression shipped.

Tests: `test_tile_buffer_capped` + `test_conv_alloc_smoke_nonpeak` -> 2 passed. The A_tile
is deterministically <= 384 MB and tile_p < P (true tiling, not full 1.21 GB im2col); the
non-peak alloc-smoke delta (0.63 GiB) stays well under the budget and under the full-im2col
size.

## Task B.6 — conv3d variant (firm win + SeedVR2 OOM target)

conv3d correctness (`test_conv3d_matches_reference`, spy active): 2 passed, max|diff| < 2e-1.

conv3d bench (M5 Max, fp16, 1x128x5x256x256, w=128x128x3x3x3, pad=1):
```
  correctness: max|diff|=0.1248 -> OK
conv3d  im2col=43.517ms  stock=69.472ms  speedup=1.60x  current_alloc_after_ours=0.21GiB (NOT peak)
```
(re-run standalone: conv3d im2col=44.675ms stock=75.019ms speedup=1.68x)

conv3d im2col is FASTER than stock conv3d (1.6-1.7x) and correct -> firm win confirmed.
Because the 64x64 baseline GEMM already beats stock conv3d, Task B.3b (register-tiling)
is NOT required. current_alloc 0.21 GiB (NOT a high-watermark; the A_tile cap is the
deterministic proof, B.8 the definitive OOM check).

## Task B.7 — guarded install + __init__.py registration

Full suite: `pytest tests/test_conv_im2col.py -v` -> 18 passed in 0.28s.
Install/fallback tests (run everywhere, not just MPS): no-op-without-flag, per-mode
idempotence (=2d then =3d installs {2,3}), grouped/dilated/CPU fallback, injected-kernel-
exception fallback -> all pass.

Flag-on grouped-fallback smoke:
```
[AppleSilicon-FP8/conv] conv im2col+matmul2d active on MPS (ranks=[2, 3], tile=384MB).
grouped-fallback ok: (1, 8, 16, 16)
```
NOTE: the plan's smoke used w=[8,8,3,3] with groups=2 which is an INVALID grouped conv
(groups=2 needs weight [8,4,3,3] for 8-ch input); the stock conv legitimately rejected it.
Corrected to w=[8,4,3,3] -> fallback delegates to stock conv, output (1,8,16,16). Proves
the fallback path (grouped conv NOT routed to the kernel).

Registered patch #18 in __init__.py (import list + install loop, after int8_linear_kernel_mps;
docstring line 18). ruff clean on _patches/conv_im2col_mps.py + tests/test_conv_im2col.py.
