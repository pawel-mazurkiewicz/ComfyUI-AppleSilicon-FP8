"""Individual MPS/FP8 patches, each exposing an idempotent install().

The README's "What it fixes" table numbers them; the top-level __init__ sets install order.
"""

__all__ = [
    "psutil_vmstat",
    "comfykitchen_fp8",
    "scaled_mm_fp8",
    "rmsnorm_mps_large",
    "flash_attn_mtl",
    "ops_bias_fp8",
    "stochastic_round_fp8",
    "tensor_to_fp8",
    "wan_blockswap_mps",
    "linear_fp8",
    "te_device_mps",
    "int_mm_mps",
    "int8_linear_mps",
    "mlx_textgen",
    "fused_norm_mps",
    "fp8_linear_kernel_mps",
]
