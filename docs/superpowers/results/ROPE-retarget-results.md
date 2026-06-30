# ROPE-retarget (patch #21b): fused RoPE retargeted onto the REAL comfy functions

**Branch:** `impl/ROPE-retarget` (off `integration/asfp8-accel`)
**File:** `_patches/rope_fast_mps.py` (extends patch #21)
**Tests:** `tests/test_rope_fast_comfy.py` (new), `tests/test_rope_fast.py` (unchanged, still green)
**Gate:** opt-in `ASFP8_ROPE_FAST=1`, default OFF, guarded, never fatal.

## Problem

Patch #21 only wrapped `comfy_kitchen.backends.eager.apply_rope{,1,_split_half,_split_half1}`.
A live probe (2026-07-01) showed the real models bypass that path almost entirely — they call
comfy's OWN rope functions:

| model | function | fires |
|---|---|---|
| Flux-2-Klein DiT | `comfy.ldm.flux.math.apply_rope` | every layer |
| Ideogram-4 DiT | `comfy.ldm.ideogram4.model.apply_rope` | every layer |
| Ideogram-4 TE | `comfy.text_encoders.llama.apply_rope` | every layer |

`comfy.ldm.ideogram4.model.apply_rope` **is** `comfy.text_encoders.llama.apply_rope` (same object,
imported by alias — verified `is` identity). `comfy_kitchen.eager` fired at most once per model, so
patch #21 was effectively inert on both.

## Conventions (verified against source)

Two **different** conventions — they are not interchangeable:

- **`comfy.ldm.flux.math.apply_rope` — interleaved 2×2.** `freqs_cis` is a single fp32 tensor of
  shape `(...,halfD,2,2)` (probe: `(1,1,8704,64,2,2)`). `_apply_rope1` reshapes `x` to
  `(...,halfD,1,2)` (interleaved pairs `2p, 2p+1`) and applies the per-pair 2×2 matrix. This is the
  **identical** convention to comfy_kitchen's `apply_rope`, so it maps onto the existing **split=0**
  kernel with no new math. (Note: stock `flux.math.apply_rope` only forwards to
  `comfy.quant_ops.ck.apply_rope` when not training — we intercept *above* that delegation.)

- **`comfy.text_encoders.llama.apply_rope` — half-split rotate_half.** `freqs_cis` is a **3-tuple**
  `(cos, sin, nsin)` (not a tensor): `cos` full-dim `D`, `sin`/`nsin` half-dim `halfD`
  (`nsin = -sin`). The op is `q*cos` then `addcmul_` of the two halves:
  `out[p] = cos[p]·x[p] + nsin[p]·x[p+halfD]`, `out[p+halfD] = sin[p]·x[p] + cos[p+halfD]·x[p+halfD]`.
  This is expressible on the **split=1** kernel by building a per-pair 2×2 table
  `[cos[:halfD], nsin, sin, cos[halfD:]]` → `[f00,f01,f10,f11]`. Bit-faithful (the real op also does
  fp32 multiply/addcmul then casts back to input dtype).

## What was done

1. `install()` now also calls `_install_comfy()`: captures the real `flux.apply_rope`,
   `flux.apply_rope1`, `llama.apply_rope` and replaces every reference to them **by object identity**
   (mirroring `mps_profile._try_wrap_rope`) — so the ideogram4 alias is rerouted automatically.
   Existing comfy_kitchen targets are untouched. Each comfy target's fallback is the **captured real
   comfy function** (not eager), so any fallback is bit-identical to stock comfy.
2. New table builder `_prep_table_llama` (tuple → fp32 `[L,halfD,4]`), new wrappers
   `_flux_apply_rope{,1}_fused` and `_llama_apply_rope_fused`, sharing a refactored `_launch`.
3. Safety: rank-4 MPS only, supported dtype, even D, fp32 table; llama requires batch/head broadcast
   (cos rows == L) and `Lq == Lk == L`. Anything else → fall back to the real comfy fn (never wrong).

## Retargeted functions

| function | convention | kernel path | matched? | tolerance |
|---|---|---|---|---|
| `comfy.ldm.flux.math.apply_rope` | interleaved 2×2 | split=0 | ✅ | bf16 atol 2e-2 / fp32 2e-5 |
| `comfy.ldm.flux.math.apply_rope1` | interleaved 2×2 | split=0 | ✅ | bf16 atol 2e-2 |
| `comfy.text_encoders.llama.apply_rope` | half-split rotate_half (cos,sin,nsin) | split=1 + built table | ✅ | fp32 2e-5 / fp16 3e-3 / bf16 2e-2 |
| `comfy.ldm.ideogram4.model.apply_rope` | (alias of llama) | split=1 | ✅ rerouted by identity | bf16 2e-2 |

**Fallbacks (correct-by-construction, tested):** llama with a true batch>1 freqs table (cos rows =
B·L ≠ L) and flux with non-broadcast (real B/H) leading table dims — both fall back to the real
comfy fn and match it. No function required a permanent fall-back due to an unmatchable convention.

## Correctness (oracle = captured REAL comfy fn; kernel proven via poison)

Each kernel test computes the oracle from the real comfy function captured before patching, asserts
allclose, and poisons the captured original so any silent fallback **raises** — proving the fused
kernel actually ran. Observed max |fused − real|:

- FLUX bf16: 1.56e-2 (≈1 bf16 ulp at magnitude ~5); fp32: 9.5e-7.
- LLAMA bf16: 3.9e-3 – 1.56e-2; fp32: 4.8e-7.

## Benchmark (verify-before-timing; kernel asserted to fire each row)

| case | dtype | real ms/call | fused ms/call | speedup | maxdiff |
|---|---|---|---|---|---|
| FLUX L=8704 H=32 D=128 (probe) | bf16 | 6.852 | 1.459 | **4.70×** | 1.56e-2 |
| FLUX L=4096 H=32 D=128 | bf16 | 3.312 | 0.692 | **4.79×** | 1.56e-2 |
| FLUX L=8704 H=32 D=128 | fp32 | 5.739 | 2.213 | 2.59× | 9.5e-7 |
| LLAMA seq=4096 H=16 hd=128 | bf16 | 1.008 | 0.392 | **2.57×** | 1.56e-2 |
| LLAMA seq=256 H=16 hd=128 | bf16 | 0.200 | 0.058 | 3.45× | 3.9e-3 |
| LLAMA seq=4096 H=16 hd=128 | fp32 | 0.952 | 0.571 | 1.67× | 4.8e-7 |

## Test counts

- `tests/test_rope_fast_comfy.py` (new): **14 passed** (flux pair/single + poison, llama param
  dtype×shape + poison, ideogram4 alias reroute, two fallback regimes).
- `tests/test_rope_fast.py` (patch #21, unchanged): **27 passed**.
- Full suite: **186 passed, 40 skipped** — no regressions.

## Environment note

The repo venv (`/Volumes/IMPERIAL SPACE/AI/ComfyUI/.venv`) does not contain the `comfy` package;
ComfyUI-desktop keeps the code tree at `/Users/pawelma/ComfyUI-Installs/ComfyUI/ComfyUI`
(`basePath` in `~/Library/Application Support/ComfyUI/config.json`). Tests add that path (overridable
via `ASFP8_COMFY_PATH`) and `importorskip` comfy, so they skip cleanly where comfy is absent. The
comfy import pulls in `comfy.model_management`, whose `psutil.virtual_memory()` call fails under the
build sandbox — runs need the sandbox-disabled path (the kernel/correctness logic itself is
unaffected).
