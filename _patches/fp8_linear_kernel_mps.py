"""Patch #20: route fp8 (e4m3) weight nn.Linear through the native fp8 Metal kernel.

Intercepts mixed_precision_ops Linear.forward BEFORE the activation is fp8-quantized and
feeds the raw activation (as half) plus the still-fp8 weight bytes to gemm_fp8_nt, then
applies the weight scale and bias in fp32. That is a semantic change, not an equivalence:
it is closer to unquantized act x dequantized weight than the path it replaces. DEFAULT
ON (ASFP8_FP8_NATIVE=off disables), and only fires on min_dim>=8192 layers.
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
    # the kernel casts bf16 activations to fp16, so an outlier above 65504 would become
    # inf; =0 skips the per-call amax check
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

    Building inside install() would block ComfyUI's startup on a ninja+clang build.
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
    """One-time parity gate: native half x fp8 vs decoded fp32. Call only after a layer
    has passed every eligibility, shape and scale gate."""
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


def _verify():
    """The contract _caps.kernel_ready expects: build, warmup, then numerics."""
    return _ensure_kernel() is not None and _self_check()


def _fp8_linear_kernel(input, qdata, scale_weight, bias):
    """half(input) x fp8 weight bytes -> f32, * scale, + bias, back to input.dtype.

    Callers MUST have validated shapes, dtype and scale via _try_fp8_kernel_forward.
    """
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

        # bail before the range guard below, which costs an amax plus a device sync
        if _kernel_tried and _kernel is None:
            return None

        # cheap rejects first, so non-fp8 layers fall through fast
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
        # shape/rank/dtype guards before the raw kernel call: the C++ never checks W
        if qdata.dim() != 2:
            return None
        if not qdata.is_mps or qdata.dtype != torch.float8_e4m3fn:   # e4m3 only
            return None
        N, K = int(qdata.shape[0]), int(qdata.shape[1])
        if input.dim() < 2 or int(input.shape[-1]) != K:
            return None
        bias = self.bias
        if bias is not None and int(bias.numel()) != N:
            return None

        scale = params.scale
        if scale.numel() not in (1, N):     # scalar or per-channel [N]
            return None

        if max(N, K) < _min_dim():
            return None

        if _range_guard_on() and input.dtype is torch.bfloat16:
            if bool((input.detach().abs().amax() > 65504).item()):
                return None

        # build + self-check only after every gate passes: this is a full extension compile
        from . import _caps
        if not _caps.kernel_ready("fp8", _verify):
            return None

        return _fp8_linear_kernel(input, qdata, scale, bias)
    except Exception as e:
        # latch off: verification already passed, so this recurs on every later layer
        from . import _caps
        if _caps._kernel_ready.get("fp8"):
            print(f"{TAG} kernel forward failed ({e!r}); using comfy's fp8 path "
                  f"for the rest of this session.")
        _caps.mark_kernel_failed("fp8")
        return None


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    from . import _caps
    if not _caps.resolve("ASFP8_FP8_NATIVE", default_on=True, cap=_caps.kernel_gate):
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
        # carry the int8 wrapper's marker over if it already wrapped the factory
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
