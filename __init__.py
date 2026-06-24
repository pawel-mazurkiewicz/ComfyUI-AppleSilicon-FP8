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
  9. WanVideo block-swap neutralizer on MPS      (block swap is CUDA-VRAM-only; breaks on MPS)
 10. F.linear FP8 operand decode on MPS         (T5 encoder + any FP8 Linear using F.linear directly)
 11. text_encoder_device CPU->MPS on MPS         (text/LLM encoders default to CPU on Apple Silicon)
 12. torch._int_mm GPU path on MPS               (INT8 models: _int_mm has no Metal kernel -> CPU-fallback freeze)
 13. int8-fast Linear via MPS bf16 GEMM           (INT8 wide-batch matmul was 3-5x too slow in fp32 _int_mm)
 14. MLX-backed Qwen3-VL TextGenerate on MPS       (Krea2 prompt-expansion: ~50s eager generate -> MLX)
 15. fp8-native Linear via Metal 4.1 matmul2d      (EXPERIMENTAL, opt-in ASFP8_FP8_EXT=1; large fp8 MLP Linears)

See README.md for details. MIT licensed.
"""

# ComfyUI looks for these on every custom node. Empty = side effects only, no nodes.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if __spec__ is not None and __spec__.parent:
    # Loaded as part of a package (normal ComfyUI startup) — install patches.
    # When imported without a real parent package (e.g. pytest --import-mode=importlib
    # imports __init__.py directly as a top-level module), __spec__.parent is an empty
    # string and relative imports would fail.  Checking __spec__.parent is CPython-
    # guaranteed behaviour (see importlib docs) and avoids inspecting ImportError
    # message strings that are implementation details liable to change.
    from ._patches import comfykitchen_fp8, linear_fp8, ops_bias_fp8, psutil_vmstat, rmsnorm_mps_large, scaled_mm_fp8, flash_attn_mtl, stochastic_round_fp8, tensor_to_fp8, wan_blockswap_mps, te_device_mps, int_mm_mps, int8_linear_mps, mlx_textgen, fp8_linear_mps

    for _patch in (psutil_vmstat, comfykitchen_fp8, scaled_mm_fp8, ops_bias_fp8, stochastic_round_fp8, tensor_to_fp8, wan_blockswap_mps, rmsnorm_mps_large, flash_attn_mtl, linear_fp8, te_device_mps, int_mm_mps, int8_linear_mps, mlx_textgen, fp8_linear_mps):
        try:
            _patch.install()
        except Exception as _e:  # never take ComfyUI down because of us
            import traceback
            print(f"[AppleSilicon-FP8] patch {_patch.__name__} failed: {_e}")
            traceback.print_exc()
