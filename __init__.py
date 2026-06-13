"""
ComfyUI-AppleSilicon-FP8

Makes FP8-quantized models (FLUX, SD3.5, Ideogram 4, ...) run on Apple Silicon
(Metal / MPS) instead of crashing, and works around a psutil bug on recent macOS
betas that otherwise kills renders mid-way.

ComfyUI imports this at startup, before any model loads, so the patches are active
for the whole session. Each patch is a no-op on machines that don't need it
(non-macOS, non-MPS, or where the relevant library isn't installed).

Patches applied:
  1. psutil.virtual_memory() vm_stat fallback   (macOS 26/27 beta crash)
  2. comfy_kitchen eager FP8 dequant/quant      (Ideogram 4 and other ck models)
  3. torch._scaled_mm FP8 on MPS                (FLUX, SD3.5, FP8 _scaled_mm path)
  4. F.rms_norm manual fp32 path on MPS         (PiD >=2048px: black image / NaN)
  5. flash_attn drop-in + fast SDPA on MPS      (mtlflashattn: 3-4x over fused SDPA,
                                                 fixes the large-attention OOM/cliff)
  6. cast_bias_weight FP8 weight+bias decode    (FP8 UNETLoader dtype: weight/bias cast crash)
  7. stochastic_rounding FP8 CPU reroute        (LoRA + FP8 base model: re-quant crash)
  8. torch.Tensor.to FP8<->float on MPS         (3rd-party fp8 Linears: WanVideo custom_linear, etc.)

See README.md for details. MIT licensed.
"""

# ComfyUI looks for these on every custom node. Empty = side effects only, no nodes.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

from ._patches import comfykitchen_fp8, ops_bias_fp8, psutil_vmstat, rmsnorm_mps_large, scaled_mm_fp8, flash_attn_mtl, stochastic_round_fp8, tensor_to_fp8

for _patch in (psutil_vmstat, comfykitchen_fp8, scaled_mm_fp8, ops_bias_fp8, stochastic_round_fp8, tensor_to_fp8, rmsnorm_mps_large, flash_attn_mtl):
    try:
        _patch.install()
    except Exception as _e:  # never take ComfyUI down because of us
        import traceback
        print(f"[AppleSilicon-FP8] patch {_patch.__name__} failed: {_e}")
        traceback.print_exc()
