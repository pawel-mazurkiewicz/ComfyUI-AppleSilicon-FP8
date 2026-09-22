"""ComfyUI-AppleSilicon-FP8: run FP8/INT8/INT4 quantized models on Apple Silicon (MPS).

ComfyUI imports this at startup and each patch installs itself, staying inert where it
isn't needed. See README.md for the patch list. MIT licensed.
"""

# ComfyUI looks for these on every custom node; empty = side effects only.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if __spec__ is not None and __spec__.parent:
    # __spec__.parent is "" when imported directly (pytest --import-mode=importlib),
    # where the relative imports below would fail.
    from ._patches import comfykitchen_fp8, linear_fp8, ops_bias_fp8, psutil_vmstat, rmsnorm_mps_large, scaled_mm_fp8, flash_attn_mtl, stochastic_round_fp8, tensor_to_fp8, wan_blockswap_mps, te_device_mps, int_mm_mps, int8_linear_mps, int8_linear_kernel_mps, fp8_linear_kernel_mps, conv_im2col_mps, mlx_textgen, fp8_mps_strided, rope_fast_mps, probe_runtime, optrace, mps_profile, fused_norm_mps

    # optional patch: not present in every checkout
    try:
        from ._patches import int4_linear_mps
    except Exception as _e:
        int4_linear_mps = None
        print(f"[AppleSilicon-FP8] int4_linear_mps not present ({_e}); skipping.", flush=True)

    # bisection switches, by patch module name; ENABLE_ONLY wins if both are set
    import os as _os
    _only = {n.strip() for n in _os.environ.get("ASFP8_ENABLE_ONLY", "").split(",") if n.strip()}
    _disabled = {n.strip() for n in _os.environ.get("ASFP8_DISABLE", "").split(",") if n.strip()}

    try:
        from ._patches import _caps
        print(f"[AppleSilicon-FP8] capabilities: {_caps.summary()}", flush=True)
    except Exception as _e:
        print(f"[AppleSilicon-FP8] capability probe failed: {_e}", flush=True)

    # optrace installs last so it sees the seams every other patch wrapped
    for _patch in (psutil_vmstat, fp8_mps_strided, comfykitchen_fp8, scaled_mm_fp8, ops_bias_fp8, stochastic_round_fp8, tensor_to_fp8, wan_blockswap_mps, rmsnorm_mps_large, fused_norm_mps, flash_attn_mtl, linear_fp8, te_device_mps, int_mm_mps, int8_linear_mps, int8_linear_kernel_mps, int4_linear_mps, fp8_linear_kernel_mps, conv_im2col_mps, mlx_textgen, rope_fast_mps, probe_runtime, optrace, mps_profile):
        if _patch is None:   # optional patch absent from this checkout
            continue
        _short = _patch.__name__.rsplit(".", 1)[-1]
        if (_only and _short not in _only) or _short in _disabled:
            print(f"[AppleSilicon-FP8] skipping {_short} (ASFP8_ENABLE_ONLY/ASFP8_DISABLE)")
            continue
        try:
            _patch.install()
        except Exception as _e:  # never take ComfyUI down because of us
            import traceback
            print(f"[AppleSilicon-FP8] patch {_patch.__name__} failed: {_e}")
            traceback.print_exc()
