# Issue F — results: route fp8 Linear through native `fp8_matmul2d_nt` (half-act × fp8-weight)

Branch: `impl/F-fp8-native-matmul` (off `main` @ `c0ff7cd`).
Machine: M5 Max, macOS 27, PyTorch 2.11 MPS, repo venv.
Status: **AUTONOMOUS portion GREEN. Feature ships DEFAULT OFF (`ASFP8_FP8_NATIVE` opt-in).
Real-model gate (Task -1) and end-to-end gate (Task 6) are HUMAN-REQUIRED and still OPEN.**

---

## (A) Autonomous tasks — GREEN on this M5

### Task 1 — loader build-gate
`_patches/fp8_ext/loader.py` now builds when **`ASFP8_FP8_EXT=1` OR `ASFP8_FP8_NATIVE=1`**.
Verified: `ASFP8_FP8_NATIVE=1` alone builds the Metal lib and exports `fp8_matmul2d_nt`.
Reset-state tests pass.

### Task 0 — SYNTHETIC numeric probe (`tests/probe_fp8_native_matmul.py`)
Native `fp8_matmul2d_nt(half_act, fp8_e4m3_weight)` vs decoded-**fp32** ground truth, on
Flux-2-Klein-class shapes. Gate: every shape `rel_native < 2e-2` AND `rel_native <= 2*rel_lut + eps`.

| M | K | N | rel_native | rel_lut | mean_abs | p99.9_rel | act\|max\| | within_2x_lut |
|---|---|---|---|---|---|---|---|---|
| 4096 | 4096 | 4096 | 7.45e-07 | 2.04e-03 | 3.90e-07 | 2.33e-05 | 1.586 | ✅ |
| 4096 | 4096 | 16384 | 9.25e-07 | 1.90e-03 | 3.91e-07 | 2.22e-05 | 1.547 | ✅ |
| 4096 | 16384 | 4096 | 2.27e-06 | 2.07e-03 | 3.03e-06 | 9.29e-05 | 1.672 | ✅ |

**WORST rel_native = 2.27e-06.** PROBE PASS. The native half-act path is ~1000× **more**
accurate than the LUT→bf16 path it replaces (fp32 accumulate + half mantissa ≫ bf16), confirming
the kernel math is correct. (Synthetic only — does NOT establish real activation range; that is Task -1.)

### Tasks 2–4 — wrapper + unit + REAL spy tests (`tests/test_fp8_linear_kernel.py`)
Wrapper module `_patches/fp8_linear_kernel_mps.py` (patch #20): shape/rank/dtype/eligibility
guards, scalar **and** per-channel `[N]` scale support, `_self_check()` ordered AFTER all
eligibility gates (never on an ineligible layer), never-fatal fallback.

pytest counts:
- **Flag OFF (`pytest tests/`):** `61 passed, 7 skipped` (kernel tests skip without the flag).
- **Flag ON (`ASFP8_FP8_NATIVE=1 pytest tests/`):** `64 passed, 4 skipped` — the 3 native/spy tests run.
- `tests/test_fp8_linear_kernel.py` alone, flag ON: **9 passed**.

Spy/parity highlights (flag ON, real Metal kernel):
- `test_native_matches_ground_truth` — native vs all-MPS fp32 ref, rel < 2e-2, output non-zero.
- `test_native_per_channel_scale` — `[N]` scale branch exercised, rel < 2e-2.
- `test_wrapper_dispatches_native_not_fallback` — **canonical SPY**: real fp8 `QuantizedTensor`
  weight (`TensorCoreFP8Layout`, built on CPU then moved to MPS), `_fp8_linear_kernel` replaced by
  a sentinel and the captured `orig_forward` set to **raise**. The wrapper returns the sentinel →
  proves `Linear.forward` dispatched the NATIVE branch, not the fallback (a fallback would ERROR).
- `test_self_check_not_run_on_ineligible` — BLOCKER-2 regression guard: self-check never runs on
  an ineligible layer.

> Note: this comfy_kitchen registers the e4m3 fp8 layout as **`TensorCoreFP8Layout`** (the eligibility
> `str(...).startswith("TensorCoreFP8")` accepts it). Direct `QuantizedTensor.from_float(..., 'TensorCoreFP8Layout')`
> on MPS raises (MPS can't cast bf16→fp8), so fp8 QTs are built on CPU then `.to('mps')` — which yields
> MPS-resident `float8_e4m3fn` qdata. The spy fixture uses that path.

### Task 5 — benchmark (`tests/bench_fp8_native_matmul.py`, verify-before-timing)
Numerical match asserted (`rel < 2e-2`) **before** any timing. Native timing includes the per-call
bf16→half cast; LUT timing reported with the per-call `decode_fp8` (the real per-step cost).

| M | K | N | native_kernel | native_e2e | lut_gemm | lut_e2e(+decode) | speedup_e2e | rel |
|---|---|---|---|---|---|---|---|---|
| 4096 | 4096 | 4096 | 2.621 ms | 2.766 ms | 2.208 ms | 2.982 ms | **1.08×** | 8.30e-07 |
| 4096 | 4096 | 16384 | 10.352 ms | 10.488 ms | 8.467 ms | 11.695 ms | **1.12×** | 8.30e-07 |
| 4096 | 16384 | 4096 | 9.633 ms | 10.469 ms | 10.261 ms | 12.927 ms | **1.23×** | 1.98e-06 |

`native_e2e` beats `lut_e2e` at every shape (1.08–1.23×) — the honest microbench win. The
hypothesized ~3.6× is an **end-to-end per-step** claim (weight DRAM traffic halved, decode kernel
removed across a 9B model) that **only Task 6 can confirm**; this microbench does not assert it.

### Task 5.1 / 5.2 — registration + regression gate
Patch #20 registered in `__init__.py` (after `int8_linear_kernel_mps`; fp8 check runs first, hands
back to int8 on non-fp8 weights) and `_patches/__init__.py` `__all__`/docs tidied. Flag-OFF import
smoke: `installed= False` (clean no-op). Full suite green both flag states (above).

---

## (B) HUMAN-REQUIRED gates — STILL OPEN (feature stays OFF until both pass)

### Task -1 (HARD PRE-WIRE GATE) — real Flux-2-Klein seam/scale/range probe
**NOT RUN** (needs Flux-2-Klein-9B loaded in ComfyUI; cannot be done in the isolated worktree).
Runnable probe script written for the human: **`dev/probe_F_flux_seam.py`** — call
`probe(model, run_one_step=<one sampler step>)` after launching ComfyUI with
`ASFP8_FP8_NATIVE=1 ASFP8_PROFILE=1` and a Flux-2-Klein fp8 workflow.

The human must confirm **all three** before patch #20 can be enabled for real renders:
1. **Seam fires?** — the eligible fp8 Linear's MRO includes `comfy.ops…mixed_precision_ops…Linear`,
   and its `FORWARD` logs **before** `torch._scaled_mm`. (If it's `fp8_ops.Linear` → re-point
   `install()`. If `manual_cast.Linear` → out of scope. If forward never precedes scaled_mm → seam
   wrong, stop.)
2. **Scale shape?** — scalar `()`/`(1,)` vs per-channel `(N,)`. Wrapper already supports both;
   the probe just records which is real (so a scalar-only assumption can't silently NO-OP every layer).
3. **Activation range?** — per-layer `input.float().abs().amax()` vs fp16 max **65504**. If all
   ≪ 65504, leave the range guard off and document the bound. If any ≳ 65504, enable
   `ASFP8_FP8_NATIVE_RANGE_GUARD=1` and record outlier layers.

### Task 6 — real Flux-2-Klein end-to-end validation
**NOT RUN** (needs a real render). After Task -1 confirms the seam: A/B the `linear` profiler
bucket (flag off vs on), confirm visual correctness (no NaN/black/garbage), and do the 3-way
real-layer accuracy comparison (current quantized vs native half-act vs fp32 ground truth) to
verify native is *closer to* fp32 than the current path. Record results here, then enable.

> **Default state:** `ASFP8_FP8_NATIVE` is **OFF**. A wrong autonomous guess is safe (never-fatal
> fallback) but is NOT validated. Enable only after Task -1 (seam/scale/range) and Task 6 (win +
> no quality regression) sign-off.
