# Issue G2 — M5 matmul2d operand dtype capability probe — RESULTS

Branch: `impl/G2`
Machine: M5 Max / macOS 27 (Darwin 27.0.0) / PyTorch 2.11.0 / Metal 4.1 (`MTLLanguageVersion4_1`)
Python: `/Volumes/IMPERIAL SPACE/AI/ComfyUI/.venv/bin/python` (3.12.11), `torch.backends.mps.is_available() == True`
Active SDK: `/Applications/Xcode-beta.app/.../MacOSX.sdk` (Metal toolchain MetalToolchain-v27.1.5194.15)
GPUCompiler: Versions/32023
Run date: 2026-06-30

> Note: per the orchestrator hard rules these results are recorded HERE, not in
> `INVESTIGATION_FACTS.md` (avoids cross-worktree merge conflicts). The probe scripts
> still target `INVESTIGATION_FACTS.md` per the plan, but that file does not exist in
> this worktree, so the scripts' existence-guard simply prints a "not appended" warning
> and writes nothing — by design.

## Prerequisites (Task 1)

- `xcrun --find metal` → `/var/run/com.apple.security.cryptexd/mnt/.../Metal.xctoolchain/usr/bin/metal` OK
- MPP header found under the ACTIVE SDK:
  `.../MacOSX.sdk/System/Library/Frameworks/MetalPerformancePrimitives.framework/Versions/A/Headers/MetalPerformancePrimitives.h`
- `ninja` available in the ComfyUI venv (`.venv/bin`)
- MPS available: True

## Task 0 — compile-only API facts (PASS = type/instantiation available under Metal 4.1)

| fact | result |
|------|--------|
| matmul2d<half> + <half,half,float> accum compiles | PASS |
| matmul2d<bfloat> + <bfloat,bfloat,float> accum compiles | PASS |
| matmul2d<float> + <float,float,float> accum compiles | PASS |
| matmul2d<signed char> + int accum compiles | PASS |
| fp8 e4m3 type available under Metal 4.1 | PASS |
| fp8 e5m2 type available under Metal 4.1 | PASS |

All six Task-0 facts PASS. `signed_char` control compiles (mandatory gate). Both fp8
macros (`__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__`, `__HAVE_METAL_FP8_E5M2_FORMAT_TYPE__`)
are defined on this SDK → no dtype is expected to surface COMPILE_SKIP.

## Task 3 — one template fix required (compile failure, NOT a capability gap)

The plan's fp8 kernel template cast the buffer pointer to `(device fp8_t*)`. The MPP
`tensor<device fp8_t, …, tensor_inline>` constructor's `data_handle_type` is
`device unsigned char*` (fp8 is addressed as bytes), so the cast was rejected:

```
error: no matching constructor ... no known conversion from 'device fp8_t *'
       to '...data_handle_type' (aka 'device unsigned char *') for 1st argument
```

Fix: pass the raw `device uchar*` (`rawA+ulong(m0)*K`) directly, mirroring the
production kernel `gemm_fp8_nt` (`_patches/fp8_ext/fp8_matmul2d.mm:96`). After the fix
both fp8 dtypes compile, run, and match exactly. This was a generated-source bug, not a
hardware/SDK limitation — Task 0 had already proven both fp8 types are available.

## S-G2 — capability matrix (the keystone output for B/D/E)

NT layout (A[M,K] @ Bᵀ, B[N,K]), BM=BN=64, NSG=4, M=N=64, K=128.
Reference = torch fp32 matmul of DECODED stored operands. Each dtype compiled in its
OWN MTLLibrary. Latency is INFORMATIONAL ONLY (launch-dominated; not a tensor-unit perf
signal). SPY: every PASS row had C ≠ all-zero AND matched the reference — a no-op/fallback
would leave C all-zero and fail.

| dtype        | compile | run | max_err  | verdict | gate (issue B / FP8 only) |
|--------------|---------|-----|----------|---------|---------------------------|
| signed_char  | OK      | OK  | 0 (exact int32) | PASS | control / spy |
| half         | OK      | OK  | 1.34e-05 | PASS | issue B conv GEMM can use fp16 operands on tensor units |
| bfloat       | OK      | OK  | 7.63e-06 | PASS | issue B conv GEMM can use bf16 operands on tensor units |
| float        | OK      | OK  | 0        | PASS | fp32 cooperative path legal (perf NOT proven here) |
| fp8_e4m3     | OK      | OK  | 0        | PASS | FP8 e4m3 operand path viable (regresses existing fp8fp8_matmul2d_nt); NOT W4A8 |
| fp8_e5m2     | OK      | OK  | 0        | PASS | FP8 e5m2 operand path viable (informational); NOT W4A8 |

### Verdict for downstream issues

- **half PASS and bfloat PASS** → issue B conv GEMM may use fp16 OR bf16 operands on the
  M5 tensor units (cooperative `matmul2d` with a float accumulator). Both are bit-correct
  vs the decoded-operand reference (max_err well under tolerance: half ≤ 0.5, bf16 ≤ 1.0).
- **float PASS** → fp32 cooperative path is LEGAL on this M5/Metal 4.1. This proves
  legality ONLY, not tensor-unit performance — a separate large-shape benchmark is needed.
- **fp8_e4m3 PASS** → confirms/regresses the existing `fp8fp8_matmul2d_nt` path; FP8 e4m3
  operands route through `matmul2d` correctly.
- **fp8_e5m2 PASS** → e5m2 is available AND usable as a matmul2d operand under Metal 4.1
  independently of e4m3 (its own macro is defined; the kernel compiles, runs, and matches).
  Informational for now.
- **W4A8 is NOT decided here.** Sub-byte int4b/uint4b/int2b operands are out of scope and
  require a separate packing + correctness probe.

## Tests (Task 4)

`pytest tests/test_probe_matmul2d_dtypes.py -v` → **8 passed in 5.80s**

- `test_signed_char_control_runs_exact` PASSED (control + spy: int8 kernel runs, matches
  int32 reference exactly, C ≠ all-zero)
- `test_every_candidate_produces_structured_row[*]` PASSED for all 6 candidates
- `test_table_generation_covers_all_candidates` PASSED (one row per candidate; "NOT W4A8" present)

No skips fired (MPS + xcrun both present). No tolerances were widened; all PASS rows are
genuine real-kernel matches.

## Caching (Task 5.2)

First build ~30 s (JIT compile of the ObjC++ host). Second run: ~6 s total (host extension
cached by `cpp_extension.load`; per-dtype Metal libraries recompiled each run, sub-ms each).

## Tooling note

`ruff` (the repo's documented linter) is not installed in the ComfyUI venv used here, nor
is `pyflakes`. All three new files parse cleanly via `ast.parse`. The one unused local
(`fp8_probe` lambda in `probe_matmul2d_task0.py`) is verbatim from the authoritative plan
and was left as-is to avoid deviating from it.
