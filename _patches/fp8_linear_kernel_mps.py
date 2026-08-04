"""Patch #20: route fp8 (e4m3) weight nn.Linear through the native fp8 Metal kernel on MPS.

Flux-2 fp8 currently runs LUT-decode(weight fp8->bf16) -> torch._scaled_mm/bf16 GEMM
(~5.5 s/step), and the activation is force-quantized bf16->fp8 upstream. This patch
intercepts the mixed_precision_ops Linear.forward BEFORE the activation is fp8-quantized,
feeds the RAW bf16 activation (cast to half) + the still-fp8 weight bytes into the
existing gemm_fp8_nt kernel (fp8_matmul2d_nt: half[M,K] x W[N,K] fp8 -> f32), applies the
weight scale (scalar OR per-channel [N]) + bias in fp32, returns bf16. Semantically this
is `input.float() @ dequant_fp8(weight).t() * scale (+bias)` -- CLOSER to unquantized-act x
dequantized-fp8-weight than the current lossy path (a semantic change, not equivalence).

Default ON (seam confirmed by the live Flux-2 probe); ASFP8_FP8_NATIVE=off disables.
Note: only fires on large layers (min_dim>=8192), so it is inert on smaller DiTs (e.g. Ideogram-4
hidden 4096). Never-fatal: any miss falls back to comfy's
original forward. The real-model seam/scale/range are gated on Task -1 (see plan).
"""
import os
import sys

import torch

TAG = "[AppleSilicon-FP8/fp8_kernel]"

_installed = False
_kernel = None
_kernel_tried = False
_self_checked = False
_self_ok = True
_MIN_DIM_DEFAULT = 8192


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _min_dim():
    return _env_int("ASFP8_FP8_NATIVE_MIN_DIM", _MIN_DIM_DEFAULT)


def _range_guard_on():
    # DEFAULT ON now that this kernel path is default-on: the kernel casts bf16
    # activations to fp16 (max finite 65504), so a bf16 outlier above that would
    # become inf and silently skew the output. The guard detects it and falls back
    # to the bit-exact decode path. Set ASFP8_FP8_NATIVE_RANGE_GUARD=0 to skip the
    # per-call amax check if you know your activations stay in fp16 range.
    return os.environ.get("ASFP8_FP8_NATIVE_RANGE_GUARD", "1").strip().lower() not in (
        "0", "off", "false", "no")


def _load_kernel():
    try:
        from .fp8_ext import loader
    except Exception as e:  # pragma: no cover
        print(f"{TAG} loader import failed: {e!r}")
        return None
    mod = loader.module()
    if mod is not None:
        try:
            mod.warmup()
        except Exception as e:
            print(f"{TAG} warmup failed; disabling: {e!r}")
            return None
    return mod


def _ensure_kernel():
    """Build the Metal extension lazily, on the FIRST layer that can actually use it.

    Building inside install() blocked ComfyUI's startup import on a synchronous
    ninja+clang build (issue: "hangs forever at startup when ninja is installed").
    Deferring it here matches scaled_mm_fp8._get_backend and keeps startup
    non-blocking; a failed build is remembered so we retry once only.
    """
    global _kernel, _kernel_tried
    if _kernel is not None or _kernel_tried:
        return _kernel
    _kernel_tried = True
    _kernel = _load_kernel()
    if _kernel is None:
        print(f"{TAG} fp8 Metal kernel unavailable; using the LUT decode path.")
    return _kernel


def _self_check():
    """One-time parity gate: native half x fp8 == decoded-fp32 within 5e-2, else disable.
    MUST be called only AFTER a layer passes every eligibility/shape/scale gate."""
    global _self_checked, _self_ok
    if _self_checked:
        return _self_ok
    _self_checked = True
    try:
        from ._common import decode_fp8
        d = _env_int("ASFP8_FP8_NATIVE_SELFCHECK_DIM", 8192)  # K=N; M fixed small
        M = 32
        g = torch.Generator().manual_seed(0)
        a = (torch.randn(M, d, generator=g) * 0.3).to(torch.bfloat16)
        w = (torch.randn(d, d, generator=g) * 0.3).to(torch.float8_e4m3fn)  # [N,K]
        ref = (a.float() @ decode_fp8(w.to("mps"), torch.float32).cpu().t()).to("mps")
        native = _kernel.fp8_matmul2d_nt(
            a.to("mps").to(torch.float16).contiguous(),
            w.to("mps").contiguous().view(torch.uint8), d)
        rel = ((native - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        _self_ok = rel < 5e-2
        if not _self_ok:
            print(f"{TAG} self-check failed (rel={rel:.4f}); using LUT path.")
    except Exception as e:
        _self_ok = False
        print(f"{TAG} self-check raised; using LUT path: {e!r}")
    return _self_ok


def _fp8_linear_kernel(input, qdata, scale_weight, bias):
    """half(input) x fp8 weight bytes -> f32, * scale (scalar or [N]), + bias, -> input.dtype.
    Callers MUST have validated shapes/dtype/scale via _try_fp8_kernel_forward first."""
    orig_shape = input.shape
    N = int(qdata.shape[0])
    x2 = input.reshape(-1, input.shape[-1])
    a_half = x2.to(torch.float16).contiguous()
    w_u8 = qdata.contiguous().view(torch.uint8)
    out = _kernel.fp8_matmul2d_nt(a_half, w_u8, N)            # [M,N] f32
    scale_f = scale_weight.to(device=out.device, dtype=torch.float32)
    n_scale = scale_f.numel()
    if n_scale == 1:
        out = out * scale_f.reshape(())                       # tensorwise dequant
    elif n_scale == N:
        out = out * scale_f.reshape(1, N)                     # per-channel dequant
    else:
        raise ValueError(f"unexpected weight scale numel={n_scale} (N={N})")
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(input.dtype).reshape(*orig_shape[:-1], N)


def _try_fp8_kernel_forward(self, input):
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        # Build already attempted and failed: bail before the range guard below,
        # which costs a full-tensor amax plus a device sync on every forward.
        if _kernel_tried and _kernel is None:
            return None

        # --- cheap rejects first (so non-fp8 / int8 / plain layers fall through fast) ---
        if not isinstance(input, torch.Tensor) or input.device.type != "mps":
            return None
        if isinstance(input, QuantizedTensor):
            return None
        if input.dtype not in (torch.bfloat16, torch.float16):
            return None

        w = self.weight
        if not isinstance(w, QuantizedTensor):
            return None
        if not str(getattr(w, "_layout_cls", "")).startswith("TensorCoreFP8"):
            return None
        if getattr(self, "_full_precision_mm", False):
            return None
        if getattr(self, "comfy_force_cast_weights", False):
            return None
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return None

        params = w._params
        if getattr(params, "transposed", False):
            return None

        qdata = w._qdata
        # --- BLOCKER: shape/rank/dtype guards BEFORE the raw kernel call (C++ never checks W) ---
        if qdata.dim() != 2:
            return None
        if not qdata.is_mps or qdata.dtype != torch.float8_e4m3fn:   # kernel decodes e4m3 only
            return None
        N, K = int(qdata.shape[0]), int(qdata.shape[1])
        if input.dim() < 2 or int(input.shape[-1]) != K:
            return None
        bias = self.bias
        if bias is not None and int(bias.numel()) != N:
            return None

        scale = params.scale
        if scale.numel() not in (1, N):     # scalar OR per-channel [N] (OQ#1, resolved by Task -1)
            return None

        if max(N, K) < _min_dim():
            return None

        # --- real-activation fp16 range guard (DEFAULT ON; =0 to skip the amax check) ---
        if _range_guard_on() and input.dtype is torch.bfloat16:
            if bool((input.detach().abs().amax() > 65504).item()):
                return None

        # --- BLOCKER: build + self-check only AFTER all gates pass, never on
        # ineligible layers (the build is a full ninja/clang extension compile) ---
        if _ensure_kernel() is None:
            return None
        if not _self_check():
            return None

        return _fp8_linear_kernel(input, qdata, scale, bias)
    except Exception as e:
        print(f"{TAG} kernel forward fell back ({e!r})")
        return None


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    # DEFAULT ON (seam confirmed via live Flux-2 probe), gated on Tier B (Metal-4 tensor
    # ops + ninja for the ObjC++ fp8 extension). Unsupported HW -> default OFF (no build
    # attempt); ASFP8_FP8_NATIVE=off force-disables, =1 forces the build attempt anyway.
    from . import _caps
    if not _caps.resolve("ASFP8_FP8_NATIVE", default_on=True, cap=_caps.tier_b_ready):
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    try:
        import comfy.ops as ops
        orig_factory = ops.mixed_precision_ops
        if getattr(orig_factory, "_asfp8_fp8_wrapped", False):
            _installed = True
            return

        def wrapped_factory(*a, **k):
            cls = orig_factory(*a, **k)
            linear_cls = getattr(cls, "Linear", None)
            if linear_cls is not None and not getattr(linear_cls, "_asfp8_fp8_patched", False):
                orig_forward = linear_cls.forward

                def forward(self, input, *args, **kwargs):
                    res = _try_fp8_kernel_forward(self, input)
                    if res is not None:
                        return res
                    return orig_forward(self, input, *args, **kwargs)

                linear_cls.forward = forward
                linear_cls._asfp8_fp8_patched = True
            return cls

        wrapped_factory._asfp8_fp8_wrapped = True
        # Preserve the int8 wrapper's marker if it already wrapped the factory.
        if getattr(orig_factory, "_asfp8_wrapped", False):
            wrapped_factory._asfp8_wrapped = True
        ops.mixed_precision_ops = wrapped_factory
    except Exception as e:
        print(f"{TAG} could not wrap mixed_precision_ops: {e!r}")
        return

    _installed = True
    print(f"{TAG} fp8 e4m3 Linear routed through native fp8 matmul2d on MPS "
          f"(half act x fp8 weight; LUT->bf16 weight decode bypassed; min_dim={_min_dim()}; "
          f"range_guard={_range_guard_on()}). Kernel builds on first fp8 layer, not now.")
