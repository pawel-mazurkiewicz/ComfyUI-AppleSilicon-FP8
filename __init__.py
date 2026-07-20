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
  2. comfy_kitchen eager FP8 + NVFP4/MXFP8 decode (Ideogram 4, LTX, other ck/mixed-precision models)
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
 14. MLX-backed TextGenerate on MPS (Qwen3-VL + Gemma3)  (Krea2 ~50s & LTX2 ~13h eager generate -> MLX)
 16. strided FP8 ops -> CPU on MPS (global)        (NVFP4/MXFP8 quant+dequant: reshape/contiguous of fp8 crashes MPS)
 17. INT8 W8A8 via bit-exact Metal kernel on MPS   (DEFAULT ON, gated on M5/Metal-4.1 + ninja; ASFP8_INT8_EXT=off
                                                    to disable: int8 matmul ~1.75x over bf16, bypasses per-step
                                                    fp32 weight dequant/un-rotation; Cider-derived kernel)
 18. fused RMSNorm+modulation+residual on MPS  (DEFAULT ON, gated on compile_shader; ASFP8_FUSED_NORM=off to disable:
                                                   one compile_shader pass
                                                   for the DiT adaLN tail; fp32+64-bit indexing also
                                                   supersedes patch #4's >2^21-row correctness fallback)
 19. conv im2col+matmul2d on MPS (ASFP8_CONV_IM2COL, conv3d DEFAULT ON / =off to disable: VAE conv via
                                                   tensor units, ~2.7x vs stock conv3d, A_tile capped at ASFP8_CONV_TILE_MB)
 20. fp8-native Linear via Metal matmul2d on MPS  (DEFAULT ON, seam confirmed; ASFP8_FP8_NATIVE=off to
                                                    disable: half act x fp8 e4m3 weight on tensor units;
                                                    bypasses per-step fp8->bf16 decode + _scaled_mm. Only
                                                    fires on min_dim>=8192 layers, so inert on Ideogram-4)
 21. fused standalone RoPE on MPS                  (DEFAULT ON, gated on compile_shader; ASFP8_ROPE_FAST=off to
                                                    disable: comfy_kitchen apply_rope / apply_rope_split_half via ONE
                                                    compile_shader kernel; ~6-17x/call over eager; fp32 math, no
                                                    Metal-4.1/M5 requirement)
 21b. fused RoPE retargeted onto real comfy funcs  (OPT-IN ASFP8_ROPE_COMFY_RETARGET=1: reroutes comfy.ldm.flux /
                                                    llama apply_rope on live models; higher-risk, off by default)
 22. INT4 ConvRot W4A16/W4A8 Linear on MPS          (DEFAULT reroute: skip comfy_kitchen's wasted activation-int4
                                                    quant, run W4A16 bf16 GEMM — fixes "int4 ~2x slower than int8"
                                                    (issue #3), brings PARITY with int8; int4's win is memory not
                                                    speed. Opt-in W4A8 fused Metal kernel via ASFP8_INT4_EXT=1)
 (#15 fp8-native F.linear retired — wrong seam; the fp8-native win is patch #3's opt-in _scaled_mm fast path)
 (#20 reserved for fp8-native matmul on a separate branch; ROPE uses #21 to avoid collision)

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
    from ._patches import comfykitchen_fp8, linear_fp8, ops_bias_fp8, psutil_vmstat, rmsnorm_mps_large, scaled_mm_fp8, flash_attn_mtl, stochastic_round_fp8, tensor_to_fp8, wan_blockswap_mps, te_device_mps, int_mm_mps, int8_linear_mps, int8_linear_kernel_mps, fp8_linear_kernel_mps, conv_im2col_mps, mlx_textgen, fp8_mps_strided, rope_fast_mps, probe_runtime, optrace, mps_profile, fused_norm_mps

    # int4_linear_mps is an optional / in-progress patch that is not present in every
    # checkout; import it defensively so a missing module never breaks the whole node at
    # startup (ImportError would otherwise abort loading before any patch installs). It
    # stays inert when absent and installs normally when the module is present.
    try:
        from ._patches import int4_linear_mps
    except Exception as _e:
        int4_linear_mps = None
        print(f"[AppleSilicon-FP8] int4_linear_mps not present ({_e}); skipping.", flush=True)

    # Bisection switches (for debugging a regression to a single patch):
    #   ASFP8_ENABLE_ONLY=psutil_vmstat,comfykitchen_fp8   install ONLY these (by module name)
    #   ASFP8_DISABLE=flash_attn_mtl,rmsnorm_mps_large     install everything EXCEPT these
    # ENABLE_ONLY wins if both are set. Names are the patch module names (last path segment).
    import os as _os
    _only = {n.strip() for n in _os.environ.get("ASFP8_ENABLE_ONLY", "").split(",") if n.strip()}
    _disabled = {n.strip() for n in _os.environ.get("ASFP8_DISABLE", "").split(",") if n.strip()}

    # Every perf patch is now DEFAULT ON but gated on the hardware/software that can run
    # it (see _patches/_caps.py). Print the detected capabilities once so users can see
    # exactly which acceleration tier is active on their machine.
    try:
        from ._patches import _caps
        print(f"[AppleSilicon-FP8] capabilities: {_caps.summary()}", flush=True)
    except Exception as _e:
        print(f"[AppleSilicon-FP8] capability probe failed: {_e}", flush=True)

    # optrace installs LAST (opt-in ASFP8_TRACE_OPS=1) so it sees every matmul the
    # model dispatches, on top of all the seams the other patches wrapped.
    for _patch in (psutil_vmstat, fp8_mps_strided, comfykitchen_fp8, scaled_mm_fp8, ops_bias_fp8, stochastic_round_fp8, tensor_to_fp8, wan_blockswap_mps, rmsnorm_mps_large, fused_norm_mps, flash_attn_mtl, linear_fp8, te_device_mps, int_mm_mps, int8_linear_mps, int8_linear_kernel_mps, int4_linear_mps, fp8_linear_kernel_mps, conv_im2col_mps, mlx_textgen, rope_fast_mps, probe_runtime, optrace, mps_profile):
        if _patch is None:   # optional patch (e.g. int4_linear_mps) absent from this checkout
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
