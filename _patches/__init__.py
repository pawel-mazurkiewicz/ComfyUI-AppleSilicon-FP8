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
    # Internal helpers (not patches):
    # "_common"             — decode_fp8, fp8_to_float_lut, FP8_DTYPES
    # "na_gemm"             — optional NA matmul2d backend (not wired into hot path)
]
