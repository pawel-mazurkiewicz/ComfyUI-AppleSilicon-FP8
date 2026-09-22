"""Individual MPS/FP8 patches, each exposing an idempotent install()."""

# Numbers match the README "What it fixes" table; install order is set in the
# top-level __init__.py, not here.
__all__ = [
    "psutil_vmstat",         # #1
    "comfykitchen_fp8",      # #2
    "scaled_mm_fp8",         # #3
    "rmsnorm_mps_large",     # #4
    "flash_attn_mtl",        # #5
    "ops_bias_fp8",          # #6
    "stochastic_round_fp8",  # #7
    "tensor_to_fp8",         # #8
    "wan_blockswap_mps",     # #9
    "linear_fp8",            # #10
    "te_device_mps",         # #11
    "int_mm_mps",            # #12
    "int8_linear_mps",       # #13
    "mlx_textgen",           # #14
    "fused_norm_mps",        # #18
    "fp8_linear_kernel_mps", # #20
]
