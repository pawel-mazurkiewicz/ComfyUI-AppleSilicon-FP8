# ComfyUI-AppleSilicon-FP8

**Run FP8-quantized models on Apple Silicon (Metal / MPS) without crashes.**

If you're on a Mac and FP8 models die with
`Trying to convert Float8_e4m3fn to the MPS backend but it does not have support for that dtype`,
`scaled_mm ... not implemented for MPS`, or your renders crash mid-way with a
`psutil ... host_statistics64 ... array not large enough` traceback — this fixes all of it.
It covers the whole pipeline that NVIDIA-targeted workflows assume "just works":
FLUX / SD3.5 / Ideogram 4, **FP8 `UNETLoader` checkpoints** (e.g. Lens), **LoRAs
applied on top of an FP8 base model**, and **third-party custom nodes that ship
their own FP8 Linear** (e.g. ComfyUI-WanVideoWrapper's text encoder and
transformer). It also fixes a couple of non-FP8 MPS bugs that hit the same
machines, including **PiD (Pixel Diffusion Decoder) producing a fully black image
at ≥2048px** and **WanVideo block swap crashing with a cpu/mps device mismatch**.

The goal is plain: the world of LoRAs, models and workflows on Civitai/etc. is
overwhelmingly trained and tuned for NVIDIA — that doesn't mean you can't run it
on Apple Silicon. This trades some peak throughput for "it actually runs."

It's a single ComfyUI custom node that applies a few targeted runtime patches at
startup. No model conversion and no Metal compilation; the only dependency is
[`mtlflashattn`](https://github.com/pawel-mazurkiewicz/mtlflashattn) (the Metal
flash-attention kernels — Apple Silicon only, installed automatically). Each
patch is a no-op on machines that don't need it.

> Tested on: Apple M-series, macOS 27 dev beta, PyTorch 2.11, Python 3.12, ComfyUI Desktop.

## What it fixes

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | `RuntimeError: host_statistics64(HOST_VM_INFO64) ... array not large enough` — renders crash partway through | psutil's prebuilt C extension doesn't match the kernel on recent macOS betas; `virtual_memory()` fails ~99% of calls, and ComfyUI calls it every node | Replace `psutil.virtual_memory()` with a `vm_stat` + `sysctl`-based equivalent that doesn't use the broken syscall |
| 2 | `TypeError: ... convert Float8_e4m3fn to the MPS backend ...` from `comfy_kitchen` (e.g. **Ideogram 4**) | comfy_kitchen's eager FP8 backend dequantizes with a plain `x.to(bfloat16)` cast, which MPS can't do from FP8 | Decode FP8 with a lookup-table + gather (bit-identical to the original, runs on GPU) |
| 3 | `scaled_mm not implemented for MPS` / FP8 cast errors from **FLUX / SD3.5** | `torch._scaled_mm` has no FP8 kernel on MPS | Patch `torch._scaled_mm` to decode FP8 → float and run a native MPS matmul |
| 4 | **PiD (Pixel Diffusion Decoder) outputs a fully black image at ≥2048px** (`RuntimeWarning: invalid value encountered in cast`) | `torch.nn.functional.rms_norm` silently returns garbage on MPS once the normalization row count exceeds ~2²² (~4.19M); PiD's pixel blocks cross that at 2048px+, producing NaN → black | Compute `rms_norm` with the exact manual fp32 formula on MPS for large row counts; the fused fast path is kept for normal sizes and all non-MPS devices |
| 5 | **Large attention SIGKILLs the render** (SeedVR2 4K DiT, long-context global attention), **or attention is slow / numerically wrong** on MPS past ~4k tokens | MPS fused `scaled_dot_product_attention` materializes the full `Lq×Lk` score matrix (memory grows `O(B·H·Lq·Lk)`) and is silently inaccurate at length; there is no flash-attention on MPS | Back `F.scaled_dot_product_attention` (and `import flash_attn`) with [`mtlflashattn`](https://github.com/pawel-mazurkiewicz/mtlflashattn): Metal flash kernels (simdgroup_matrix / M5 TensorOps) that never form the score matrix and run **3–4× faster than fused SDPA** at length. Gated so small attention stays on stock |
| 6 | `TypeError: ... convert Float8_e4m3fn to the MPS backend ...` from an **FP8 `UNETLoader` checkpoint** (e.g. Lens, FLUX fp8) at sampling time | ComfyUI's `manual_cast` layers store weight **and bias** as raw FP8 and cast them up per forward; MPS can cast neither *to* nor *from* FP8 on-device (the bias crashes first, the weight would crash next) | Take over `comfy.ops.cast_bias_weight` on the plain MPS path and LUT-decode weight + bias to the compute dtype (QuantizedTensor params routed via `dequantize()`) |
| 7 | `TypeError: ... convert Float8_e4m3fn ...` when applying a **LoRA on top of an FP8 base model** | After patching the float weight, ComfyUI re-quantizes it back to FP8 via `stochastic_rounding`, which does a float→FP8 cast that MPS can't | Route the FP8 re-quant through CPU (where the cast works), then move the FP8 result back to MPS |
| 8 | `TypeError: ... convert Float8_e4m3fn ...` from a **custom node's own FP8 Linear** (e.g. WanVideoWrapper `custom_linear.py`, T5 encoder) | These bypass `comfy.ops` and cast FP8 weights/bias at runtime with a plain Python `.to(input)`; MPS can't cast to/from FP8 | Wrap `torch.Tensor.to` so FP8↔float conversions on MPS go through the LUT decode (FP8→float) or CPU (float→FP8); everything else takes a tight fast path. *(Catches Python-level `.to()`; FP8 promotion inside C++ ops like `F.linear` is out of scope — see Scope.)* |
| 9 | `RuntimeError: Expected all tensors to be on the same device, but found ... mps:0 and cpu!` in **WanVideoSampler** | WanVideo **block swap** offloads transformer blocks to CPU and streams them back per-step, syncing the async copy with CUDA events that don't hold on MPS — so a block's params (e.g. `self.modulation`) are still on CPU when it runs | On MPS, neutralize block swap: wrap `WanModel.forward` to clear the offload flags and make every block resident on the compute device first. Memory is unified on Apple Silicon, so block swap saves nothing here anyway |

### How the FP8 trick works

PyTorch's MPS backend has no 8-bit float type, so you can't cast to/from
`float8_e4m3fn` / `float8_e5m2` on the GPU. But you *can* move FP8 tensors from
CPU to MPS, bit-view them as `uint8`, and gather/index on MPS. So we build a
256-entry table mapping every FP8 byte to its float value (decoded once on CPU,
where the cast works), move it to the GPU, and decode any FP8 tensor with
`lut[x.view(uint8)]`. This is **bit-exact** with a real FP8→float cast and runs
entirely on the GPU. Matmuls then use MPS's native (fast) float matmul.

## Install

### ComfyUI-Manager (easiest)
Manager → *Install via Git URL* →
`https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8`

### Manual
```bash
cd <your ComfyUI>/custom_nodes
git clone https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8
```
Then restart ComfyUI.

## Verify it's active

At startup you'll see (only the lines relevant to your machine):

```
[AppleSilicon-FP8/psutil] psutil.virtual_memory() is broken on this OS — installed vm_stat fallback (...).
[AppleSilicon-FP8/comfy_kitchen] patched comfy_kitchen eager FP8 dequantize/quantize for MPS.
[AppleSilicon-FP8/scaled_mm] torch._scaled_mm FP8 now runs on MPS via LUT decode + native matmul.
[AppleSilicon-FP8/ops_bias] cast_bias_weight FP8 weight+bias LUT-decoded to compute dtype on MPS.
[AppleSilicon-FP8/stochastic_round] stochastic_rounding FP8 re-quant routed via CPU on MPS.
[AppleSilicon-FP8/tensor_to] torch.Tensor.to FP8<->float routed via LUT/CPU on MPS.
[AppleSilicon-FP8/wan_blockswap] armed; will neutralize WanVideo block swap on MPS when it loads.
[AppleSilicon-FP8/rmsnorm] F.rms_norm uses manual fp32 path on MPS for >2^21 rows (PiD black-image fix).
[AppleSilicon-FP8/flash] F.scaled_dot_product_attention -> mtlflashattn on MPS (correctness>=4096 tok, fast-tier>=1024 tok, oom>=12 GB).
```

## Notes & caveats

- **Accuracy:** the FP8 decode is bit-exact; results match a CUDA/CPU FP8 run
  within normal quantization noise.
- **Speed:** patches lean on MPS's native float matmul rather than a custom Metal
  kernel — correctness and zero-setup over peak throughput. It's plenty usable;
  it is not a hand-tuned fused FP8 kernel.
- **The psutil fix is macOS-only and self-disabling.** It only activates if
  `psutil.virtual_memory()` actually fails a startup probe (a clear majority of
  calls) on your machine — which only happens on the affected macOS betas. On any
  healthy/older macOS it detects nothing wrong and leaves psutil completely
  untouched, so it cannot break lower systems. You can override the auto-detection:

  | `APPLESILICON_FP8_PSUTIL` | Behaviour |
  |---|---|
  | unset / `auto` (default) | Activate only if psutil is actually broken here |
  | `off` / `0` | Never touch psutil |
  | `on` / `force` / `1` | Always use the `vm_stat` fallback |

  Set it in your shell/launch environment, e.g. `APPLESILICON_FP8_PSUTIL=off`.
- **comfy_kitchen / `_scaled_mm` / `cast_bias_weight` / `Tensor.to` FP8 patches**
  only act when FP8 is genuinely involved and MPS is in play; CUDA, CPU, and all
  non-FP8 tensors take an unchanged fast path. The decode is bit-exact.
- **FP8 on Apple Silicon is a compatibility, not a speed, feature.** MPS has no
  real FP8 compute, so every FP8 path decodes to bf16 before the matmul anyway —
  you keep FP8's *storage* savings but pay a per-use decode and run at
  bf16-equivalent speed. If you have the RAM, a bf16 checkpoint avoids the decode
  tax entirely and is usually faster.
- **WanVideo block swap is neutralized on MPS (patch #9).** Block swap exists to
  fit models into scarce NVIDIA VRAM; Apple Silicon memory is unified, so it saves
  nothing and its CUDA-event-synced streaming breaks on MPS. The patch makes the
  model run fully resident regardless of what a downloaded workflow configured.
  Disable with `ASFP8_NEUTRALIZE_BLOCKSWAP=off`.
- **The `rms_norm` fix is MPS-only and row-count gated.** It swaps in a manual
  fp32 `rms_norm` only on MPS and only when the normalization row count exceeds
  2²¹ (~2.1M) — the regime where the fused kernel is wrong. Everything else (all
  non-MPS devices, all normal-sized tensors) keeps the fast fused path untouched.
- **Flash attention / SDPA (patch #5) is MPS-only and gated.** It backs
  `F.scaled_dot_product_attention` and `import flash_attn` with `mtlflashattn`,
  but only reroutes when it helps: correctness (max seq ≥ 4096 tokens, where stock
  MPS SDPA is silently wrong), a fast TensorOps tier (max seq ≥ 1024), or an OOM
  rescue (would-be score matrix ≥ 12 GB). Small attention stays on stock fused
  SDPA, and any unsupported case or kernel error falls straight back — it never
  crashes the render. Tunables:

  | Env var | Behaviour |
  |---|---|
  | `MTLFLASHATTN_SDPA` = `off` | Disable the SDPA reroute (legacy alias: `APPLESILICON_FP8_SDPA=off`) |
  | `MTLFLASHATTN_SHIM` = `off` | Disable the `flash_attn` drop-in shim |
  | `MTLFLASHATTN_SDPA_MIN_SEQ` (4096) | Correctness gate: route at/above this sequence length |
  | `MTLFLASHATTN_SDPA_FAST_MIN_SEQ` (1024) | Speed gate: route when a fast TensorOps tier is available |
  | `MTLFLASHATTN_SDPA_MIN_GB` (12) | OOM-rescue gate (legacy alias: `APPLESILICON_FP8_SDPA_MIN_GB`) |

  Requires the `mtlflashattn` package (installed automatically); if it's missing,
  patch #5 logs a one-line install hint and disables itself.

## Scope

This is a "make it work on Mac" compatibility layer, not a performance library.
It targets the specific gaps that block FP8 diffusion models on MPS today. If a
model hits a *different* unsupported op (e.g. some `nvfp4` / `mxfp8` paths), it
may surface a new error — open an issue with the traceback.

**Known gap:** FP8 type-promotion that happens *inside* a C++ op (e.g. calling
`F.linear` directly with a raw FP8 weight, as WanVideo's T5 text encoder does)
isn't visible to the `Tensor.to` shim (patch #8), since no Python `.to()` is
involved. For now, set such a node's quantization to a float dtype (e.g. the
WanVideo T5 encoder → `disabled`) — on MPS that's the better choice anyway. An
`F.linear` shim to close this is on the roadmap.

## License

MIT — see [LICENSE](LICENSE).
