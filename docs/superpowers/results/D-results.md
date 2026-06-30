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
