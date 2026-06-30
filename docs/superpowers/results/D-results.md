# Issue D — empirical results

## Environment
- M5 Max, macOS 27, PyTorch 2.11.0 MPS, Metal 4.1 (MTLLanguageVersion4_1)
- Python: /Volumes/IMPERIAL SPACE/AI/ComfyUI/.venv/bin/python (3.12.11)
- torch.backends.mps.is_available() = True; hasattr(torch.mps,'compile_shader') = True

## P0 — Metal transcendental compile probe (dev/probe_metal_transcendentals.py)
Command: `PYTORCH_ENABLE_MPS_FALLBACK=1 python dev/probe_metal_transcendentals.py`

```
[precise.exp_tanh] COMPILE OK
[fast.exp_tanh] COMPILE OK
[erf.unqualified] COMPILE FAIL: SyntaxError: use of undeclared identifier 'erf'
[erf.metal] COMPILE FAIL: SyntaxError: no member named 'erf' in namespace 'metal'
PROBE COMPLETE
```

### Decision (per plan P0 decision rules)
- `precise::exp` / `precise::tanh` COMPILE OK -> use `precise::` spelling in the store (default plan).
- `erf` FAILS both unqualified and `metal::erf` -> **REMOVE `act=3` ("gelu", erf) ENTIRELY**:
  dropped from `_ACT`, store kernels (no `act==3u` branch), pybind docs, public Python contract,
  and A1/B1 parametrize lists. Supported acts: `{none:0, silu:1, gelu_tanh:2}`.
- Therefore A1 parametrize = ["silu", "gelu_tanh"]; A6 expects 8 passed (2 acts x 2 bias x 2 M).

## Phase A — point activation (SiLU / GELU-tanh)

### Two empirical findings that shaped the implementation
1. **Metal `precise::tanh` is low-accuracy.** With the plan's literal GELU-tanh formula
   `0.5*x*(1+precise::tanh(z))`, the kernel diverged from `F.gelu(approximate='tanh')` by up to
   **162 bf16 ulp** (14.7% of elements off by >1 ulp). `precise::exp`, by contrast, makes SiLU
   bit-exact-to-1-ulp. Fix: compute GELU-tanh via the exp/sigmoid identity
   `gelu(x) = x / (1 + exp(-2c(x + 0.044715 x^3)))`, `2c = 1.5957691216057308`. After this,
   GELU-tanh matches torch in absolute terms (max|d| ~0.0156, near-zero epsilon ~7.6e-6 only).
2. **The plan's `rtol=2e-3` is below bf16 precision.** One bf16 ulp is relative `2**-7 ~= 7.8e-3`.
   The plan's rationale assumed outputs ~1.0, but this test data produces outputs up to ~15, where
   one ulp = 0.0625. A *perfect* bf16 kernel (even SiLU, provably 1-ulp) cannot pass `rtol=2e-3`.
   Tolerance therefore set to `atol=2e-3, rtol=8e-3` (= one bf16 ulp). This still rejects real bugs:
   the buggy `precise::tanh` GELU had a 0.0156 absolute diff at small `ref`, which exceeds atol.

### A6/A7 test run (ASFP8_INT8_EXT=1, real Metal kernel, spy guard active)
`pytest -k "fused_activation or rejects_unknown_act or fallback_applies or bit_exact"` -> **11 passed**:
- 8x `test_int8_linear_fused_activation_matches_reference` (silu/gelu_tanh x bias x M={1,256})
  — spy guard (`_orig_int8_linear` -> raise) proves the REAL fused kernel ran. SiLU within 1 ulp;
  GELU-tanh within atol/rtol.
- `test_wrapper_fallback_applies_activation` (off-MPS fallback still applies silu).
- `test_wrapper_rejects_unknown_act` (ValueError on typo).
- `test_kernel_matches_original_bit_exact` (A7 regression guard, still `torch.equal` on no-act path).

### A8 benchmark (dev/bench_fused_activation.py, M=4096 K=1536 N=6144)
```
correctness: max|d|=0.03125   (== 1 bf16 ulp; allclose atol=2e-3 rtol=8e-3 PASS)
separate=1.035ms fused=0.824ms speedup=1.26x
```
Fused SiLU epilogue is 1.26x faster than (no-act kernel + torch.silu); no slowdown.
