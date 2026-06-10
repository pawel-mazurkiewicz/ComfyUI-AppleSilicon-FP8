# ComfyUI-AppleSilicon-FP8

**Run FP8-quantized models on Apple Silicon (Metal / MPS) without crashes.**

If you're on a Mac and FLUX, SD3.5, or Ideogram 4 die with errors like
`Trying to convert Float8_e4m3fn to the MPS backend but it does not have support for that dtype`,
`scaled_mm ... not implemented for MPS`, or your renders crash mid-way with a
`psutil ... host_statistics64 ... array not large enough` traceback — this fixes all of it.
It also fixes a couple of non-FP8 MPS bugs that hit the same machines, including
**PiD (Pixel Diffusion Decoder) producing a fully black image at ≥2048px**.

It's a single ComfyUI custom node that applies a few targeted runtime patches at
startup. No model conversion, no Metal compilation, no extra dependencies. Each
patch is a no-op on machines that don't need it.

> Tested on: Apple M-series, macOS 27 dev beta, PyTorch 2.11, Python 3.12, ComfyUI Desktop.

## What it fixes

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | `RuntimeError: host_statistics64(HOST_VM_INFO64) ... array not large enough` — renders crash partway through | psutil's prebuilt C extension doesn't match the kernel on recent macOS betas; `virtual_memory()` fails ~99% of calls, and ComfyUI calls it every node | Replace `psutil.virtual_memory()` with a `vm_stat` + `sysctl`-based equivalent that doesn't use the broken syscall |
| 2 | `TypeError: ... convert Float8_e4m3fn to the MPS backend ...` from `comfy_kitchen` (e.g. **Ideogram 4**) | comfy_kitchen's eager FP8 backend dequantizes with a plain `x.to(bfloat16)` cast, which MPS can't do from FP8 | Decode FP8 with a lookup-table + gather (bit-identical to the original, runs on GPU) |
| 3 | `scaled_mm not implemented for MPS` / FP8 cast errors from **FLUX / SD3.5** | `torch._scaled_mm` has no FP8 kernel on MPS | Patch `torch._scaled_mm` to decode FP8 → float and run a native MPS matmul |
| 4 | **PiD (Pixel Diffusion Decoder) outputs a fully black image at ≥2048px** (`RuntimeWarning: invalid value encountered in cast`) | `torch.nn.functional.rms_norm` silently returns garbage on MPS once the normalization row count exceeds ~2²² (~4.19M); PiD's pixel blocks cross that at 2048px+, producing NaN → black | Compute `rms_norm` with the exact manual fp32 formula on MPS for large row counts; the fused fast path is kept for normal sizes and all non-MPS devices |

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
[AppleSilicon-FP8/rmsnorm] F.rms_norm uses manual fp32 path on MPS for >2^21 rows (PiD black-image fix).
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
- **comfy_kitchen / `_scaled_mm` patches** only touch the MPS + FP8 path; CUDA and
  CPU behavior is completely unchanged.
- **The `rms_norm` fix is MPS-only and row-count gated.** It swaps in a manual
  fp32 `rms_norm` only on MPS and only when the normalization row count exceeds
  2²¹ (~2.1M) — the regime where the fused kernel is wrong. Everything else (all
  non-MPS devices, all normal-sized tensors) keeps the fast fused path untouched.

## Scope

This is a "make it work on Mac" compatibility layer, not a performance library.
It targets the specific gaps that block FP8 diffusion models on MPS today. If a
model hits a *different* unsupported op (e.g. some `nvfp4` / `mxfp8` paths), it
may surface a new error — open an issue with the traceback.

## License

MIT — see [LICENSE](LICENSE).
