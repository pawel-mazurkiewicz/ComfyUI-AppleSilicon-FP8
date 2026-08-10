<img width="2560" height="1280" alt="ComfyUI_00004_" src="https://github.com/user-attachments/assets/b2c78072-5f28-4f65-8ee5-72dc55b5db0b" />


# ComfyUI-AppleSilicon-FP8

**Run FP8- and INT8-quantized models on Apple Silicon (Metal / MPS) — without crashes, and faster.**

## TL;DR

- **What:** one ComfyUI custom node that patches MPS at startup so FP8/INT8
  diffusion models (FLUX, SD3.5, Ideogram 4, Krea2, …) **run on Apple Silicon
  instead of crashing** — plus Metal flash-attention and **bit-exact fp8/int8
  Metal matmul + fused-norm + RoPE kernels** for speed. No model conversion; every
  patch is a no-op on machines that don't need it.
- **Install:** ComfyUI Manager → search *AppleSilicon-FP8*; or `git clone` into
  `ComfyUI/custom_nodes/` and `pip install -r requirements.txt`. The one required
  dependency (`mtlflashattn`) installs automatically. Restart ComfyUI.
- **Speedups are ON by default — but gated on the hardware/software that can run
  them.** Every acceleration patch self-detects capability at startup and only
  activates where it's supported; everywhere else it stays inert (nothing to
  configure, nothing breaks). The startup log prints a one-line capability summary
  so you can see exactly which tier is active on your machine. The heavier matmul
  kernels need Metal-4 tensor ops — established by a runtime shader-compile probe,
  in practice an **M5** — plus `ninja` to build, so on M1–M4 they normally don't
  engage. The **int8/int4** kernels compile against **Metal 4.0** (macOS 26+); only
  the **fp8** kernel requires **Metal 4.1** (**macOS 27**), for its fp8 format type. The per-patch switches below (`ASFP8_FP8_EXT`, `ASFP8_INT8_EXT`
  and friends) take `off` to force a patch off and `1` to force it past the
  capability probe; forcing on buys only a *build attempt*, since callers still
  apply their own eligibility checks (dtype, layout, size thresholds), so a
  forced kernel that builds may still never be reached. The numeric settings
  (`ASFP8_EXT_BUILD_TIMEOUT`, `ASFP8_FP8_EXT_MIN_DIM`) tune behaviour and
  bypass nothing.

## Quick start — by machine

**Any Apple Silicon, any macOS that runs current ComfyUI + PyTorch 2.11 (M1 →
M5):** install the node and restart ComfyUI. That's it — everything in the
[What it fixes](#what-it-fixes) table is automatic: FP8/INT8 models load and
render, flash-attention accelerates long-context attention, and the psutil /
PiD-black-image / WanVideo-blockswap crashes are gone. No environment variables,
no Metal 4.1, no specific GPU generation needed. This is all most people want.

The heavier *native* matmul kernels (`ASFP8_FP8_EXT`, `ASFP8_FP8_NATIVE`,
`ASFP8_INT8_EXT`) are on by default too, but only actually engage where two axes
are satisfied:

- **OS / Metal:** the **int8/int4** kernels compile against **Metal 4.0**, so they
  work on **macOS 26** as well as 27. The **fp8** kernel compiles against **Metal
  4.1** and so needs **macOS 27**. Older macOS → the capability probe fails and the
  kernel stays inert. Needs Xcode command-line tools (the Metal compiler) and
  `ninja` to build the ObjC++ extension.
- **GPU:** the **int8** kernel needs Metal 4 cooperative TensorOps; the **fp8**
  kernel needs Metal 4.1's fp8 format type — in practice an **M5**.

**How the probe actually decides.** There is no chip-model check anywhere in the
code. The gate is `tier_b_ready()`: it (a) tries to compile a small
`mpp::tensor_ops::matmul2d` Metal shader via `torch.mps.compile_shader` and (b)
looks for `ninja`. Both must succeed. Because (a) depends on your macOS, Metal
SDK and PyTorch build as much as on the GPU, the answer is not strictly "M5 =
yes, M1–M4 = no": an M5 on some stacks probes `tensor_ops=no`, and an M4 on a
recent macOS + PyTorch nightly has probed `tensor_ops=yes`. The startup log's
`capabilities:` line prints exactly what your machine resolved to — trust that
over the chip name.

**M1 / M2 / M3 / M4 — nothing to do.** These normally fail the tensor-ops probe,
so the M5-class matmul kernels stay inert and never build. You still get the full
compatibility layer plus the compile_shader speedups that *don't* need Metal 4.1
(fused RMSNorm #18, fused RoPE #21): FP8 matmuls run accelerated via bf16 decode
on `simdgroup_matrix`, and int8 runs comfy's (fixed) weight-only path.

**M5 / M5 Pro / M5 Max on a recent macOS — just install the build toolchain and
the kernels turn themselves on:**

```bash
xcode-select --install                 # if not already installed (Metal toolchain)
pip install ninja                      # or:  pip install 'comfyui-applesilicon-fp8[kernels]'
# ...then launch ComfyUI. The int8 (#17) and fp8-native (#3/#20) kernels detect
# M5 + Metal 4.1 + ninja and activate automatically — no env vars required.
```

Each kernel is compiled lazily, on the **first model layer that can use it** —
never during ComfyUI's startup import. The build prints a `compiling the ... Metal
kernel` line first so it can't be mistaken for a freeze, and is abandoned after
`ASFP8_EXT_BUILD_TIMEOUT` seconds (default 600) so a wedged toolchain degrades to
"kernel unavailable" instead of hanging. A build lock left behind by a killed
ComfyUI is cleared automatically once it is older than that timeout, so a
force-quit mid-build can't wedge later runs. Any failure —
too-old macOS, missing `ninja`, build error — silently falls back, so the defaults
are safe everywhere. To force a kernel off on a supported machine, set its env var
to `off` (e.g. `ASFP8_INT8_EXT=off`); to force a build attempt anyway, set it
to `1`.

It started as an FP8 compatibility layer and has grown into a broader Apple
Silicon quantization layer: it keeps FP8 **and INT8** diffusion models running on
MPS, and ships **bit-exact Metal matmul kernels** (on by default where the
hardware supports them) that beat the default path (fp8-native `_scaled_mm`
~1.2–2.1×; int8 W8A8 ~1.9× on the matmul, ~24% faster end-to-end on Krea2 convrot
int8mixed) plus compile_shader fused kernels (RMSNorm, RoPE) that need no Metal 4.1.

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

It also handles **INT8** checkpoints (e.g. **Krea2 convrot int8mixed**): comfy's
int8 path runs weight-only on MPS (dequantize the int8 weight to bf16, and for
convrot models un-rotate the whole weight in fp32, every step), so — on an M5 with
the toolchain — the int8 kernel switches it to the W8A8 path the format was
designed for — rotate the *activation* online, then a real int8×int8 matmul —
which is both correct and faster.

It's a single ComfyUI custom node that applies a few targeted runtime patches at
startup. No model conversion. FP8 matmuls decode (bit-exact) to bf16 and run on
Apple's matrix units (Neural Accelerators on M5+, simdgroup_matrix on M1–M4); the
native matmul kernels go further and run fp8/int8 matmuls directly on those units
where the hardware allows. The only required dependency is
[`mtlflashattn`](https://github.com/pawel-mazurkiewicz/mtlflashattn) (the Metal
flash-attention kernels — Apple Silicon only, installed automatically); the
native matmul kernels additionally need `ninja` (the `[kernels]` extra) and stay
inert without it. Each patch is a no-op on machines that don't need it, and every
speedup is gated on a startup capability probe (see the banner in
[Verify it's active](#verify-its-active)).

> Tested on: Apple M-series, macOS 27 dev beta, PyTorch 2.11, Python 3.12, ComfyUI Desktop.

## What it fixes

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | `RuntimeError: host_statistics64(HOST_VM_INFO64) ... array not large enough` — renders crash partway through | psutil's prebuilt C extension doesn't match the kernel on recent macOS betas; `virtual_memory()` fails ~99% of calls, and ComfyUI calls it every node | Replace `psutil.virtual_memory()` with a `vm_stat` + `sysctl`-based equivalent that doesn't use the broken syscall |
| 2 | `TypeError: ... convert Float8_e4m3fn ...` (e.g. **Ideogram 4**) or `RuntimeError: Undefined type Float8_e4m3fn` from a **mixed-precision / NVFP4 checkpoint** (e.g. **LTX**'s Gemma3 text encoder) | comfy_kitchen's eager backend dequantizes per-tensor FP8 with a plain `x.to(bfloat16)` cast MPS can't do; its newer microscaling layouts (NVFP4/MXFP8) unswizzle FP8 block-scales with a reshape-after-transpose, and MPS can't make a non-contiguous FP8 tensor contiguous | Decode per-tensor FP8 with a lookup-table + gather (bit-identical, on GPU); route the NVFP4/MXFP8 block-scale dequant (`dequantize_nvfp4`/`dequantize_mxfp8`) through the CPU and move the float result back (bit-exact) |
| 3 | `scaled_mm not implemented for MPS` / FP8 cast errors from **FLUX / SD3.5** (and Krea2 `fp8_scaled`) | `torch._scaled_mm` has no FP8 kernel on MPS | Patch `torch._scaled_mm` to decode FP8 → float and run a native MPS matmul. **Both seams are wrapped:** torch ≥ 2.11 adds the public `F.scaled_mm` (`aten::_scaled_mm_v2`) and comfy_kitchen prefers it on a bare `hasattr`, with no backend check — so wrapping only `torch._scaled_mm` leaves the patch attached to a function nothing calls, and fp8 silently drops to a ~2.5× slower dequant detour (issue [#19](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8/issues/19)). **On by default where supported (M5 + Metal 4.1 + `ninja`; `ASFP8_FP8_EXT=off` to disable):** large fp8×fp8 matmuls (the real scaled-fp8 seam) skip both bf16 decodes and run a Metal 4.1 fp8-native `matmul2d` instead — bit-exact, ~1.2–2.1× faster; stays inert / falls back automatically on unsupported machines |
| 4 | **PiD (Pixel Diffusion Decoder) outputs a fully black image at ≥2048px** (`RuntimeWarning: invalid value encountered in cast`) | `torch.nn.functional.rms_norm` silently returns garbage on MPS once the normalization row count exceeds ~2²² (~4.19M); PiD's pixel blocks cross that at 2048px+, producing NaN → black | Compute `rms_norm` with the exact manual fp32 formula on MPS for large row counts; the fused fast path is kept for normal sizes and all non-MPS devices |
| 5 | **Large attention SIGKILLs the render** (SeedVR2 4K DiT, long-context global attention), **or attention is slow / numerically wrong** on MPS past ~4k tokens | MPS fused `scaled_dot_product_attention` materializes the full `Lq×Lk` score matrix (memory grows `O(B·H·Lq·Lk)`) and is silently inaccurate at length; there is no flash-attention on MPS | Back `F.scaled_dot_product_attention` (and `import flash_attn`) with [`mtlflashattn`](https://github.com/pawel-mazurkiewicz/mtlflashattn): Metal flash kernels (simdgroup_matrix / M5 TensorOps) that never form the score matrix and run **3–4× faster than fused SDPA** at length. Gated so small attention stays on stock |
| 6 | `TypeError: ... convert Float8_e4m3fn to the MPS backend ...` from an **FP8 `UNETLoader` checkpoint** (e.g. Lens, FLUX fp8) at sampling time | ComfyUI's `manual_cast` layers store weight **and bias** as raw FP8 and cast them up per forward; MPS can cast neither *to* nor *from* FP8 on-device (the bias crashes first, the weight would crash next) | Take over `comfy.ops.cast_bias_weight` on the plain MPS path and LUT-decode weight + bias to the compute dtype (QuantizedTensor params routed via `dequantize()`) |
| 7 | `TypeError: ... convert Float8_e4m3fn ...` when applying a **LoRA on top of an FP8 base model** | After patching the float weight, ComfyUI re-quantizes it back to FP8 via `stochastic_rounding`, which does a float→FP8 cast that MPS can't | Route the FP8 re-quant through CPU (where the cast works), then move the FP8 result back to MPS |
| 8 | `TypeError: ... convert Float8_e4m3fn ...` from a **custom node's own FP8 Linear** (e.g. WanVideoWrapper `custom_linear.py`) | These bypass `comfy.ops` and cast FP8 weights/bias at runtime with a plain Python `.to(input)`; MPS can't cast to/from FP8 | Wrap `torch.Tensor.to` so FP8↔float conversions on MPS go through the LUT decode (FP8→float) or CPU (float→FP8); everything else takes a tight fast path |
| 9 | `RuntimeError: Expected all tensors to be on the same device, but found ... mps:0 and cpu!` in **WanVideoSampler** | WanVideo **block swap** offloads transformer blocks to CPU and streams them back per-step, syncing the async copy with CUDA events that don't hold on MPS — so a block's params (e.g. `self.modulation`) are still on CPU when it runs | On MPS, neutralize block swap: wrap `WanModel.forward` to clear the offload flags and make every block resident on the compute device first. Memory is unified on Apple Silicon, so block swap saves nothing here anyway |
| 10 | `RuntimeError: MPS device does not support linear for non-float weights` from **any `nn.Linear` (or custom Linear) with an FP8 weight** (e.g. WanVideo T5 encoder, any FP8-quantized transformer layer) | `nn.Linear.forward` calls `F.linear`, which is a C++ op; FP8 dtype-promotion happens inside C++ before any Python `.to()` is called, so patch #8 can't intercept it | Wrap `torch.nn.functional.linear` so FP8 input/weight/bias are decoded to the compute dtype before calling the original kernel. Covers `nn.Linear` automatically since its `forward` calls `F.linear` |
| 11 | **Text/LLM encoders run on CPU** on Apple Silicon (`CLIP/text encoder model load device: cpu`), painfully slow for autoregressive encoders like **Krea2** that *generate* hundreds of tokens (~1.7s/token on CPU) | ComfyUI hardcodes `vram_state = SHARED` on MPS (unified memory), and `text_encoder_device()` only returns the GPU for `--gpu-only` or HIGH/NORMAL_VRAM — `SHARED` falls through to `return cpu`, so every text encoder lands on CPU | On MPS only, wrap `text_encoder_device()` so its CPU default is redirected to the Metal device (the same load device `--gpu-only` picks, but scoped to the encoder — offload/VAE/intermediate are left untouched). No-op on CUDA/CPU and under `--cpu`/`--gpu-only` |
| 12 | **INT8-quantized models hang at 0/N with the GPU idle** (e.g. **Krea2 `int8_mixed`** via the `int8-fast` node) — looks frozen, never crashes | INT8 Linears do their matmul with `torch._int_mm` (int8×int8→int32), which has **no Metal kernel**; with `PYTORCH_ENABLE_MPS_FALLBACK` on (ComfyUI sets it) the op silently bounces both operands to the CPU and back, per Linear per layer per step | On MPS, route `torch._int_mm` through a GPU float32 matmul instead of the CPU fallback (int8→float32 casts natively; sums stay well under int32 range and are rescaled to bf16 downstream, so it's bit-exact in practice). Non-MPS keeps the native integer kernel |
| 13 | **INT8 models run, but ~3-5× too slow** on MPS (e.g. Krea2 `int8_mixed` at ~140 s/step) | The `int8-fast` node's wide-batch path (image diffusion = thousands of tokens) quantizes activations to int8 and matmuls via `torch._int_mm` — which on MPS is float32 (patch #12), losing bf16 throughput and doubling the working set on top of a multi-GB model | On MPS, route int8-fast's `int8_forward_dynamic[_per_row]` through its own small-batch path: dequantize the int8 weight to bf16 and use MPS's native (double-buffered) bf16 GEMM. ~3.5-4.7× faster on FLUX-shaped Linears, equal-or-better accuracy (weight-only int8). Patched via a post-import hook since int8-fast loads after this node |
| 14 | **MLX-backed Qwen3-VL `TextGenerate`** — Krea2 prompt-expansion runs an eager autoregressive Qwen3-VL-4B loop (~50 s on MPS) | The generation loop runs token-by-token inside a Python `for` loop using PyTorch MPS ops; there is no batched Metal kernel for autoregressive decoding, so each forward pass pays per-token dispatch overhead (~1 s/token) | On MPS, with `mlx-vlm` installed, route the generation loop through MLX (`mlx-community/Qwen3-VL-4B-Instruct-4bit`): MLX's native autoregressive engine amortises dispatch cost with a fused Metal decode loop and KV-cache on the GPU. Text-only; conditioning encode is untouched. Eager fallback if MLX is absent or errors. |
| 16 | `RuntimeError: Undefined type Float8_e4m3fn` from **NVFP4/MXFP8 mixed-precision quant or dequant** (e.g. on-the-fly nvfp4 re-quant of an LTX text encoder, or loading such a checkpoint) | MPS has no copy kernel for FP8 with non-trivial strides: materialising a *non-contiguous* fp8 tensor (`reshape` after `transpose`, `.contiguous()`/`clone()` of a strided fp8 view) crashes. comfy's block-scale swizzles (`comfy.float.to_blocked`/`from_blocked`, comfy_kitchen's NVFP4/MXFP8 dequant) hit this in many places | Fix the primitive once: wrap the materialising Tensor methods (`reshape`/`contiguous`/`clone`) so that, only for FP8 tensors on MPS, the op falls back to CPU and the result returns to the device. Bit-exact (pure data rearrangement); no-op for every non-FP8 tensor. Covers all current and future call sites. Disable with `ASFP8_DISABLE=fp8_mps_strided` |
| 17 | **INT8 models run correctly but leave performance on the table** on MPS (e.g. **Krea2 convrot int8mixed**): comfy's int8 path is weight-only (W8A16), so every step dequantizes the int8 weight to bf16 — and for convrot checkpoints un-rotates the *entire* weight in fp32 — which dominates GPU time | comfy disables the real int8 matmul on non-CUDA (no `torch._int_mm` Metal kernel), so int8 only buys storage; the per-step fp32 weight dequant + Hadamard un-rotation is pure overhead, and naively enabling comfy's W8A8 forward pre-quantizes activations *tensorwise before* convrot (≈16% error → garbage) | **On by default where supported (M5 + Metal 4.0 / macOS 26+ + `ninja`; `ASFP8_INT8_EXT=off` to disable):** route int8 convrot Linears through the W8A8 path the format intends — rotate the *activation* online, per-row quantize it, then a **bit-exact INT8×INT8→INT32 Metal kernel** (Metal 4 cooperative TensorOps, M5+) — and skip comfy's weight dequant/un-rotation entirely. The kernel runs ~1.85× over bf16 / ~7× over the fp32 `_int_mm` fallback (with the per-row rescale + bias **fused into the store epilogue** so the int32 product never hits global memory); ~24% faster Krea2 renders at matching quality. Kernel ported from [Cider](https://github.com/Mininglamp-AI/cider) (MIT). Stays inert / falls back to comfy's path where unsupported |
| 18 | **DiT adaLN tail (RMSNorm + modulation + residual) is several separate MPS passes** | Each of rmsnorm, the `(1+scale)·x+shift` affine, and the residual add is its own kernel launch + full read/write of the activation | **On by default where supported (any MPS with `compile_shader`; `ASFP8_FUSED_NORM=off` to disable):** fuse the whole tail into **one** `compile_shader` pass (fp32 reduction, 64-bit indexing — also supersedes patch #4's >2²¹-row correctness fallback). No Metal 4.1 / M5 needed |
| 19 | **VAE / SeedVR2 conv3d decode is slow (or OOMs non-tiled) on MPS** | Stock MPS `conv3d` doesn't use the tensor units and materialises large intermediates | **On by default where supported (M5 / Metal 4.1; `ASFP8_CONV_IM2COL=off` to disable):** run conv3d as im2col + `matmul2d` on the tensor units (~2.7× vs stock conv3d, ~31% faster SeedVR2), with the patch buffer capped at `ASFP8_CONV_TILE_MB`. conv2d stays off unless opted in (`=2d`/`=2d,3d`); falls back per-conv on any unsupported shape |
| 20 | **FP8 `mixed_precision_ops` Linear decodes fp8→bf16 every step** (Ideogram-4-style eager fp8 checkpoints) | The weight is stored fp8 but each forward LUT-decodes it to bf16 before the matmul | **On by default where supported (M5 + Metal 4.1 + `ninja`; `ASFP8_FP8_NATIVE=off` to disable):** route wide fp8 Linears (min_dim ≥ 8192) through a native fp8-e4m3 `matmul2d` — half activation × fp8 weight on the tensor units — bypassing the per-step decode. Seam confirmed via a live Flux-2 probe; falls back per-call otherwise |
| 21 | **Rotary position embedding is a chain of small MPS ops per attention block** | `apply_rope` / `apply_rope_split_half` do the interleave/rotate as several elementwise kernels | **On by default where supported (any MPS with `compile_shader`; `ASFP8_ROPE_FAST=off` to disable):** fuse `comfy_kitchen`'s `apply_rope` / `apply_rope_split_half` into **one** `compile_shader` kernel (~6–17×/call over eager; fp32 math, no Metal 4.1/M5). |
| 22 | **INT4 ConvRot models run ~2× slower than INT8 on MPS** (e.g. Krea2 int4 convrot — GitHub [#3](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8/issues/3)) | comfy_kitchen's eager `convrot_w4a4` quantizes + packs the *activation* to int4, then unpacks **both** operands back to bf16 for a float GEMM — the int4 activation quant only ever fed a CUDA int4 MMA, so on MPS it's pure wasted work (measured **2.24× int8**) | On MPS, reroute ConvRot W4A4 Linears to a **W4A16 fast path** (default, `compile_shader`-free, comfy_kitchen-gated): skip the wasted activation quant, dequantize the packed int4 weight and run MPS's bf16 GEMM — faster *and* more accurate (15.8% vs 22.5% rel err). Brings int4 to **parity with int8** — see the caveat below: **int4's benefit is memory, not speed**. Opt-in W4A8 fused Metal kernel via `ASFP8_INT4_EXT=1` (M5 / Metal 4.0 + `ninja`) |
| 23 | `RuntimeError: Undefined type Float8_e4m3fn` while **loading an NVFP4 checkpoint** (e.g. LTX-2.3 — GitHub [#8](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8/issues/8)) | The NVFP4 *quantize* direction swizzles its fp8 block-scales through `to_blocked`, which pads with `padded[:rows, :cols] = input_matrix` — a strided fp8 copy MPS has no kernel for. Fires when `in_features % 64 != 0`, on any path that re-quantizes (LoRA/patch application, lowvram-streamed layers). Patch #2 covered only the *dequantize* direction | Route fp8 `to_blocked` through the CPU and return the result to the device — pure data movement, so bit-exact. Applied to both copies: `comfy.float` (the `stochastic_rounding > 0` path) and `comfy_kitchen.float_utils` (the default branch, since `comfy.quant_ops` passes `stochastic_rounding=0`) |

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

At startup you'll see a capability summary (which acceleration tier is active on
your machine), then only the patch lines relevant to it:

```
[AppleSilicon-FP8] capabilities: macOS=27.0, torch=2.11.0, mps=yes, compile_shader=yes, tensor_ops(M5/Metal4)=yes, ninja=yes
[AppleSilicon-FP8/psutil] psutil.virtual_memory() is broken on this OS — installed vm_stat fallback (...).
[AppleSilicon-FP8/comfy_kitchen] patched comfy_kitchen eager FP8 dequantize/quantize for MPS.
[AppleSilicon-FP8/scaled_mm] torch._scaled_mm FP8 on MPS via LUT decode + bf16 matrix-unit matmul. F.scaled_mm (v2 seam) wrapped too.
[AppleSilicon-FP8/ops_bias] cast_bias_weight FP8 weight+bias LUT-decoded to compute dtype on MPS.
[AppleSilicon-FP8/stochastic_round] stochastic_rounding FP8 re-quant routed via CPU on MPS.
[AppleSilicon-FP8/tensor_to] torch.Tensor.to FP8<->float routed via LUT/CPU on MPS.
[AppleSilicon-FP8/wan_blockswap] armed; will neutralize WanVideo block swap on MPS when it loads.
[AppleSilicon-FP8/rmsnorm] F.rms_norm uses manual fp32 path on MPS for >2^21 rows (PiD black-image fix).
[AppleSilicon-FP8/fused-norm] fused rmsnorm+modulation kernel active on MPS (F.rms_norm rerouted; supersedes the >2^21-row fp32 fallback).
[AppleSilicon-FP8/flash] F.scaled_dot_product_attention -> mtlflashattn on MPS (correctness>=4096 tok, fast-tier>=1024 tok, oom>=12 GB).
[AppleSilicon-FP8/linear_fp8] F.linear FP8 operands decoded to compute dtype on MPS.
[AppleSilicon-FP8/te_device] text_encoder_device redirected CPU->MPS on Apple Silicon (LLM/CLIP encoders run on GPU).
[AppleSilicon-FP8/int_mm] torch._int_mm runs on GPU (float32) on MPS instead of falling back to CPU (INT8 models).
[AppleSilicon-FP8/int8_linear] int8-fast wide-batch matmul routed via MPS native bf16 GEMM (was fp32 _int_mm).
[AppleSilicon-FP8/int8_kernel] INT8 convrot Linear routed through bit-exact Metal kernel on MPS (clean W8A8; weight-only fp32 dequant/un-rotation bypassed).
[AppleSilicon-FP8/int4_linear] ConvRot W4A4 (int4) Linear on MPS -> W4A16 rotated-basis fast path.
[AppleSilicon-FP8/conv] conv im2col+matmul2d active on MPS (ranks=[3], tile=384MB).
[AppleSilicon-FP8/rope-fast] fused RoPE active on MPS (eager apply_rope/apply_rope_split_half rerouted; interleaved + split-half; fp32 math, no Metal-4.1/M5 requirement).
[AppleSilicon-FP8/mlx_textgen] TextGenerate routed through MLX on Apple Silicon (qwen3vl_4b -> mlx-community/Qwen3-VL-4B-Instruct-4bit; gemma3_12b -> mlx-community/gemma-3-12b-it-qat-abliterated-lm-4bit).
```

> **Note:** the first line is the capability probe — `tensor_ops(M5/Metal4)` and
> `ninja` both `yes` is what unlocks the fp8/int8 matmul kernels (#3/#17/#20). The
> `conv` / `fused-norm` / `rope-fast` lines appear wherever their tier is
> supported (conv needs Metal 4; fused-norm and rope-fast need only
> `compile_shader`). The `int8_kernel` / fp8-native lines appear on an M5 with the
> toolchain + `ninja`; elsewhere those patches silently stay inert and int8 stays
> on comfy's weight-only path. The `mlx_textgen` line appears only when `mlx-vlm`
> is installed (`pip install 'comfyui-applesilicon-fp8[mlx]'`); otherwise patch #14
> no-ops.

## Notes & caveats

- **Every Metal kernel proves itself before it is used.** The capability probe
  says the machine has Metal-4 tensor ops; it does not say a *particular* kernel
  builds, and those are different questions — a macOS update once tightened a
  template constraint that broke only the int8 shader while the probe stayed
  green ([#13](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8/issues/13),
  [#14](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8/issues/14)).
  So on first eligible layer each kernel builds its own extension, runs a warmup
  dispatch, and checks its numerics against a reference; only then is it enabled.
  The verdict — including failure — is remembered for the session, so a kernel
  that cannot run costs one attempt and then falls back silently, rather than
  retrying per layer and ending up **slower than not having it**.
- **Accuracy:** the FP8 decode is bit-exact; results match a CUDA/CPU FP8 run
  within normal quantization noise.
- **Speed:** FP8 operands decode (bit-exact) to **bf16**, so the matmul runs on the
  matrix units (M5+ Neural Accelerators / M1–M4 simdgroup_matrix) instead of the
  scalar f32 path. FP8 itself is not matrix-accelerated on Metal (emulated), so
  bf16 decode is the fast route — measured ~4× faster than f32 on M5 Max across
  diffusion-shaped GEMMs.
- **If a kernel stops compiling after an OS update, that is expected-ish — and
  it will not slow you down.** The MPP tensor-ops headers ship with **macOS**,
  not Xcode, so a point release can reject a shader that built yesterday with no
  toolchain change on your side. The node degrades to comfy's own path (correct
  results, just slower) and says so once. Worth knowing if you go looking:
  `xcrun metal` compiles against the **Xcode SDK**, while the runtime compiles
  against **`/System`**, so an offline check is only trustworthy when your Xcode
  SDK is at least as new as the OS headers — otherwise it can compile a shader
  the runtime will refuse. (Thanks to @rsamerica for pinning that down.)
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
- **The default FP8 path is compatibility; the native kernels add speed where the
  hardware allows.** With no capable GPU, MPS has no real FP8 compute, so every FP8
  path decodes to bf16 before the matmul — you keep FP8's *storage* savings but pay
  a per-use decode and run at bf16-equivalent speed. If you have the RAM, a bf16
  checkpoint avoids the decode tax and is usually faster. **The native kernels
  change this** for fp8 and int8 (see `ASFP8_FP8_EXT` / `ASFP8_FP8_NATIVE` /
  `ASFP8_INT8_EXT` below): on an M5 with `ninja` (fp8 needs Metal 4.1 / macOS 27;
  int8 only Metal 4.0 / macOS 26+) they run the quantized
  matmul natively on the matrix units and beat the bf16 baseline — and they're **on
  by default**, gated on that capability probe, so they self-enable there and stay
  inert everywhere else.
- **fp8-native matmul (patch #3) is ON by default, gated on M5 + Metal 4.1 +
  `ninja`.** Large fp8×fp8 matmuls on the **`torch._scaled_mm` seam** — the path
  FLUX / SD3.5 / Krea2 `fp8_scaled` checkpoints actually take (both operands fp8 +
  scales) — are routed through a JIT-built Metal 4.1 `matmul2d` that reads fp8
  operands directly (no bf16 materialization, fp32 accumulate), then applies scales
  in fp32: **bit-exact** vs the decode path, ~1.2–2.1× faster across diffusion
  shapes. (An earlier `F.linear` seam, patch #15, was retired — ComfyUI's fp8
  checkpoints route through `_scaled_mm`, not `F.linear`, so it never fired.)

  It builds an ObjC++ extension on first use (needs the toolchain + `ninja`); the
  capability probe skips it entirely on unsupported machines, and any build/parity
  failure falls back automatically to the decode path — so the node works
  everywhere. Set `ASFP8_FP8_EXT=off` to force it off on a capable machine, or `=1`
  to force a build attempt regardless of the probe.

  | Env var | Behaviour |
  |---|---|
  | `ASFP8_FP8_EXT` (default on where capable) | fp8-native kernel at the `_scaled_mm` seam. Unset → on iff M5 + Metal 4.1 + `ninja`; `off`/`0` → force off; `1`/`on` → force the build attempt. |
  | `ASFP8_FP8_EXT_MIN_DIM` (8192) | Route to the fp8 kernel only if `max(K, N)` (weight dims) ≥ this. |
  | `ASFP8_FP8_NATIVE` (default on where capable) | Same gating, for the fp8 `mixed_precision_ops` Linear seam (patch #20, min_dim ≥ 8192). `off` to disable. |
- **int8 W8A8 native matmul (patch #17) is ON by default, gated on M5 + Metal 4.1
  + `ninja`.** Where supported, int8 convrot
  Linears (e.g. **Krea2 convrot int8mixed**) run the W8A8 path the format intends:
  rotate the activation online, per-row quantize it, then a **bit-exact
  INT8×INT8→INT32 Metal kernel** (Metal 4 cooperative TensorOps, **M5+ only**),
  bypassing comfy's per-step fp32 weight dequant + Hadamard un-rotation. Measured
  ~1.85× over the bf16 GEMM and ~7× over the fp32 `_int_mm` fallback; ~24% faster
  Krea2 renders at matching quality (clean W8A8 ≈ 1.3% rel-err, on par with the
  weight-only path). For the tensorwise-scale bf16 case the per-row rescale and
  bias add are **fused into the kernel's store epilogue** (Cider's
  `w8a8_matmul_fused_dequant`), so the int32 product is never written to global
  memory — bit-identical to the unfused path, ~1.2–1.65× faster per call. The
  kernel is **ported from
  [Cider](https://github.com/Mininglamp-AI/cider)** (Mininglamp, MIT) — its
  CUTLASS-style register-tiled `matmul2d` is what gets int8 past the fp16-class
  throughput ceiling a naive cooperative-tensor kernel hits. Builds an ObjC++
  extension on first use; the capability probe skips it on unsupported machines,
  and any build/parity failure (incl. pre-M5, no toolchain, no `ninja`) falls back
  to comfy's weight-only int8 path automatically.

  | Env var | Behaviour |
  |---|---|
  | `ASFP8_INT8_EXT` (default on where capable) | int8 W8A8 Metal kernel for int8 convrot Linears. Unset → on iff M5 + Metal 4.1 + `ninja`; `off`/`0` → force off; `1`/`on` → force the build attempt. |
  | `ASFP8_EXT_BUILD_TIMEOUT` (default `600`) | Seconds to wait for a Metal extension build (int8 #17, fp8 #3/#20, int4 #22) before giving up and falling back. An abandoned build lock is cleared once it is **twice** this old (twice the 600 s default when set to `0`), so a build that is merely slow is never disturbed. `0` disables the watchdog and waits indefinitely — not recommended, a wedged toolchain then hangs the render. |

- **fused RMSNorm (#18) and fused RoPE (#21) are ON by default on any MPS with
  `compile_shader`** — no Metal 4.1 / M5 needed. Patch #18 fuses the DiT adaLN tail
  (rmsnorm + `(1+scale)·x+shift` + residual) into one `compile_shader` pass and
  supersedes patch #4's >2²¹-row fallback; patch #21 fuses `comfy_kitchen`'s
  `apply_rope` / `apply_rope_split_half` into one kernel (~6–17×/call). Both fall
  back per-call on any unsupported shape.

  | Env var | Behaviour |
  |---|---|
  | `ASFP8_FUSED_NORM` (default on where capable) | Fused rmsnorm+modulation+residual kernel (#18). `off`/`0` → disable; `1` → force on. |
  | `ASFP8_ROPE_FAST` (default on where capable) | Fused standalone RoPE kernel (#21). `off`/`0` → disable; `1` → force on. |
- **conv im2col (#19) is ON by default for conv3d on M5 / Metal 4.1.** VAE / SeedVR2
  conv3d runs as im2col + `matmul2d` on the tensor units (~2.7× vs stock).

  | Env var | Behaviour |
  |---|---|
  | `ASFP8_CONV_IM2COL` (default `3d` where capable) | `off`/`0` → disable; `3d` (default) → conv3d only; `2d` / `2d,3d` / `1` → include conv2d. Unset + non-M5 → inert. |
  | `ASFP8_CONV_TILE_MB` (384) | Cap the im2col patch buffer (MB) so large convs tile instead of OOMing. |
- **INT4 ConvRot (patch #22) — set expectations: parity with INT8, and the win is
  MEMORY, not speed.** This patch *fixes* the reported "int4 is ~2× slower than int8"
  bug (GitHub #3) by skipping comfy_kitchen's wasted activation-int4 quant, but it
  does **not** make int4 faster than int8. On M5 the int4 tensor dtype (`int4b`) has
  **no cooperative-input matmul intrinsics** (verified in the Metal 4.1 headers), so
  it's structurally capped at int8's throughput — measured **parity within ±3%** at
  diffusion shapes, int4 edging ahead only as the token count grows. int4's real
  benefit is **resident memory** (~6.5 GB vs ~12.3 GB for int8), at int8-parity
  speed. The default path is a `compile_shader`-free W4A16 reroute (weight-only,
  comfy_kitchen-gated, inert off-MPS / on older comfy_kitchen); the opt-in W4A8
  fused Metal kernel adds nothing at compute-bound shapes and is off by default.

  **Experimental / honest caveats:** the kernels are bit-exact vs a torch reference
  in headless tests, but int4 has **not yet been visually validated on a full
  render**, and there is a known **unresolved ~1.7× gap** between live-ComfyUI int4
  and the headless harness at matching shapes that nobody has root-caused. Treat int4
  as experimental; prefer int8 unless you specifically need the memory saving.

  | Env var | Behaviour |
  |---|---|
  | `ASFP8_INT4_EXT` (default **off**) | Opt-in W4A8 fused int4 Metal kernel (M5 / Metal 4.1 + `ninja`). Off → the default W4A16 reroute handles int4. The default reroute itself has no switch (it's a compatibility path, like patch #13); disable everything int4 with `ASFP8_DISABLE=int4_linear_mps`. |
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

First and foremost a "make it work on Mac" compatibility layer: it targets the
specific gaps that block FP8 / INT8 diffusion models on MPS today. On top of that
it adds **bit-exact acceleration kernels** (fp8-native, int8 W8A8, int4 W4A16/W4A8,
fused RMSNorm, fused RoPE, conv im2col) for the hot seams — **on by default, but
gated by a startup capability probe** so each only activates on the hardware/software
that can run it (and any env var above forces a patch on or off). (int4's heavy W4A8
kernel is the exception — off by default; see the int4 caveat above.) It is not a general
performance library. If a model hits a *different* unsupported op (e.g. some
`nvfp4` / `mxfp8` compute paths), it may surface a new error — open an issue with
the traceback.

**Global switches** (debugging / overrides): `ASFP8_DISABLE=patch1,patch2` installs
everything *except* the named patch modules; `ASFP8_ENABLE_ONLY=patch1,patch2`
installs *only* those (bisection). Names are the module names in the startup log
(e.g. `fused_norm_mps`, `rope_fast_mps`). Per-patch env vars (above) are the
preferred way to toggle a single acceleration.

## Credits

- [`mtlflashattn`](https://github.com/pawel-mazurkiewicz/mtlflashattn) — Metal
  flash-attention kernels (patch #5).
- [Cider](https://github.com/Mininglamp-AI/cider) (Mininglamp, MIT) — the int8
  cooperative-TensorOps `matmul2d` kernel that patch #17's bit-exact
  INT8×INT8→INT32 GEMM (and its fused `w8a8_matmul_fused_dequant` epilogue) is
  ported from.
- [@rsamerica](https://github.com/rsamerica) — the INT8 retry-storm report
  ([#13](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8/issues/13)),
  independently root-causing the address-space qualifier behind it, and
  confirming the fix on their own rig.

## License

MIT — see [LICENSE](LICENSE).
