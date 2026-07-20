"""Individual MPS/FP8 patches, each exposing an idempotent install()."""

# Patch numbers match the README "What it fixes" table (#1–#10).
# Install order (the tuple in the top-level __init__.py) differs from patch
# number order; the comments here use patch numbers, not install positions.
__all__ = [
    "psutil_vmstat",        # patch #1  — psutil vm_stat fallback (macOS beta crash)
    "comfykitchen_fp8",     # patch #2  — comfy_kitchen eager FP8 dequant (Ideogram 4)
    "scaled_mm_fp8",        # patch #3  — torch._scaled_mm FP8 → bf16 + matrix-unit matmul
    "rmsnorm_mps_large",    # patch #4  — F.rms_norm large-row fp32 path (PiD black image)
    "flash_attn_mtl",       # patch #5  — mtlflashattn SDPA (large attention OOM/correctness)
    "ops_bias_fp8",         # patch #6  — cast_bias_weight FP8 weight+bias decode (UNETLoader)
    "stochastic_round_fp8", # patch #7  — stochastic_rounding FP8 re-quant via CPU (LoRA)
    "tensor_to_fp8",        # patch #8  — torch.Tensor.to FP8↔float shim (Python-level .to())
    "wan_blockswap_mps",    # patch #9  — WanModel.forward block-swap neutralizer
    "linear_fp8",           # patch #10 — F.linear / nn.Linear FP8 operand decode
    "te_device_mps",        # patch #11 — text_encoder_device CPU→MPS on Apple Silicon
    "int_mm_mps",           # patch #12 — torch._int_mm on GPU on MPS (INT8 models, no CPU fallback)
    "int8_linear_mps",      # patch #13 — int8-fast wide-batch Linear via MPS native bf16 GEMM
    "mlx_textgen",          # patch #14 — MLX-backed Qwen3-VL TextGenerate (prompt expansion)
    "fused_norm_mps",       # patch #18 — fused rmsnorm+modulation+residual MPS kernel (DEFAULT ON, compile_shader-gated)
    "fp8_linear_kernel_mps", # patch #20 — fp8 e4m3 Linear via native Metal matmul2d (DEFAULT ON, Tier-B gated; ASFP8_FP8_NATIVE=off to disable)
    # patch #15 (fp8_linear_mps, opt-in fp8-native *F.linear*) RETIRED — wrong seam:
    #   real ComfyUI fp8 routes through torch._scaled_mm (QuantizedTensor hides the fp8
    #   dtype so the F.linear gate never fired), and weight-only fp8 only won at tiny M.
    #   The fp8-native win lives in patch #3's _scaled_mm fast path (ASFP8_FP8_EXT=1).
    #   NOTE: patch #20 (fp8_linear_kernel_mps) is a *different, newer* seam — it wraps
    #   mixed_precision_ops.Linear.forward (the same factory seam patch #17 uses for int8),
    #   intercepting BEFORE the activation is fp8-quantized, so the F.linear-gate problem
    #   that retired #15 does not apply. DEFAULT ON (seam confirmed via a live Flux-2 probe),
    #   gated on Tier B (M5/Metal-4.1 + ninja); ASFP8_FP8_NATIVE=off disables.
    # Internal helpers (not patches):
    # "_common"             — decode_fp8, fp8_to_float_lut, FP8_DTYPES
    # "na_gemm"             — optional NA matmul2d backend (not wired into hot path)
    # "_mlx_qwen3vl"        — MLX Qwen3-VL load/generate backend for patch #14
    # "fp8_ext"             — Metal 4.1 fp8 matmul2d ObjC++ extension (patch #3 fp8-native scaled_mm backend)
]
