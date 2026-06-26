"""Patch: route comfy_kitchen INT8 convrot layers through our bit-exact Metal kernel.

Background
----------
comfy ships int8 (`int8_tensorwise`, e.g. Krea2 convrot int8mixed) but **disables
the real int8 compute path on every non-CUDA platform**: `comfy.quant_ops` sets
``QUANT_ALGOS["int8_tensorwise"]["quantize_input"] = False`` (no CUDA -> no fast
int8 matmul). So on MPS every step instead runs **weight-only W8A16**: it
dequantizes the full int8 weight to bf16 *and un-rotates the whole convrot weight
in fp32* per layer per step (~50% of GPU time), then a bf16 ``F.linear``.

Naively flipping ``quantize_input = True`` does NOT work: comfy's forward then
pre-quantizes the activation **tensorwise** (a single scalar scale over the whole
[M,K] activation) via ``QuantizedTensor.from_float`` *before* convrot can spread
the outliers, then re-quantizes inside ``int8_linear``. Measured: that pre-quant
adds ~16% error (17x worse than W8A16) -> structured garbage. (That is exactly why
comfy disabled it for int8.)

What works (measured ~1.3% err, on par with W8A16's ~0.9%): the **clean W8A8**
path -- feed ``comfy_kitchen.int8_linear`` the *raw bf16* activation so it does the
convrot rotation first and *then* per-row quantization, then an int8xint8 matmul.

So this patch wraps the ``mixed_precision_ops`` factory and replaces the int8
``Linear.forward`` so that, on MPS, eligible int8 layers compute via our
kernel-backed ``int8_linear`` on the raw bf16 input -- skipping both comfy's lossy
pre-quant and the per-step fp32 weight dequant/un-rotation. The matmul runs on our
bit-exact INT8xINT8->INT32 Metal kernel (~102 TF/s, ~1.75x over bf16). For the
tensorwise-scale bf16 case (Krea2's int8mixed), the rescale ``float(C)*row_scale[m]``
and bias add are **fused into the kernel's store epilogue** (Cider's
``w8a8_matmul_fused_dequant``), so the int32 product never round-trips through
global memory -- bit-identical to the chunked path, ~1.2-1.65x faster per call.
Everything else (other quant formats, transposed weights, LoRA weight/bias
functions, non-MPS) falls back to comfy's original forward unchanged.

Opt-in: only active when ``ASFP8_INT8_EXT=1`` (the kernel build flag).
"""

import os
import sys

import torch

TAG = "[AppleSilicon-FP8/int8_kernel]"

_installed = False
_orig_int8_linear = None
_kernel = None


def _load_kernel():
    try:
        from .int8_ext import loader
    except Exception as e:  # pragma: no cover - import wiring
        print(f"{TAG} loader import failed: {e!r}")
        return None
    return loader.module()


def _int8_linear_kernel(
    x,
    weight,
    weight_scale,
    bias=None,
    out_dtype=torch.bfloat16,
    convrot=False,
    convrot_groupsize=256,
):
    """Kernel-backed drop-in for comfy_kitchen eager int8_linear.

    Semantics match the original exactly (optional convrot activation rotation,
    per-row int8 activation quant, int8xint8->int32 matmul, chunked rescale by
    weight_scale*row_scale, optional bias). Only the matmul backend differs (our
    NT kernel vs torch._int_mm) and the weight is used in its stored [N,K] layout.
    """
    if (
        _kernel is None
        or x.device.type != "mps"
        or weight.device.type != "mps"
        or weight.dtype != torch.int8
    ):
        return _orig_int8_linear(
            x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize
        )

    from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    if convrot:
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, h, convrot_groupsize)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    x_8, x_scale = quantize_int8_rowwise(x_2d)
    weight_scale = weight_scale.view(-1)

    # Fused fast path: when the rescale is purely per-row (tensorwise weight
    # scale, the int8_tensorwise case) and the output is bf16, fold
    # float(C)*row_scale[m] (+bias) straight into the kernel's store epilogue so
    # the int32 product never round-trips through global memory. Bit-identical to
    # the chunked path below (verified equal across convrot/bias/3D/M=1).
    if (
        out_dtype == torch.bfloat16
        and weight_scale.numel() == 1
        and hasattr(_kernel, "i8_matmul2d_nt_fused")
    ):
        row_scale = (weight_scale.float() * x_scale.reshape(-1).float()).contiguous()
        bias_arg = bias.to(torch.bfloat16) if bias is not None else None
        result = _kernel.i8_matmul2d_nt_fused(
            x_8.contiguous(), weight.contiguous(), row_scale, bias_arg
        )
        return result.reshape(*orig_shape[:-1], weight.shape[0])

    # C[M,N] int32 = x_8[M,K] @ weight[N,K]^T  (NT: weight in stored layout).
    result = _kernel.i8_matmul2d_nt(x_8.contiguous(), weight.contiguous())

    m, n = result.shape
    chunk_size = max(1, min(m, 256 * 1024 * 1024 // (n * 4)))
    scaled_parts = []
    for i in range(0, m, chunk_size):
        end_i = min(i + chunk_size, m)
        chunk = result[i:end_i].float()
        chunk = chunk * (weight_scale * x_scale[i:end_i])
        scaled_parts.append(chunk.to(out_dtype))
    result = torch.cat(scaled_parts, dim=0)

    if bias is not None:
        result = result + bias.to(device=result.device, dtype=result.dtype)

    return result.reshape(*orig_shape[:-1], weight.shape[0])


def _try_int8_kernel_forward(self, input):
    """Return the layer output via the kernel W8A8 path, or None to fall back.

    Eligible iff: kernel built; input is a plain MPS Tensor (not a QuantizedTensor);
    weight is a TensorWiseINT8Layout QuantizedTensor; not full-precision-mm; no
    force-cast; no LoRA weight/bias functions; weight not logically transposed.
    """
    if _kernel is None:
        return None
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        if not isinstance(input, torch.Tensor) or input.device.type != "mps":
            return None
        if isinstance(input, QuantizedTensor):
            return None

        w = self.weight
        if not isinstance(w, QuantizedTensor) or getattr(w, "_layout_cls", None) != "TensorWiseINT8Layout":
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
        scale = params.scale
        convrot = getattr(params, "convrot", False)
        gs = getattr(params, "convrot_groupsize", 256)
        bias = self.bias

        return _int8_linear_kernel(input, qdata, scale, bias, input.dtype, convrot, gs)
    except Exception as e:
        # Never take down a render; fall back to comfy's original forward.
        print(f"{TAG} kernel forward fell back ({e!r})")
        return None


def install():
    global _installed, _orig_int8_linear, _kernel
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if os.environ.get("ASFP8_INT8_EXT") != "1":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    _kernel = _load_kernel()
    if _kernel is None:
        return  # loader already explained why

    # Capture the original eager int8_linear for the fallback inside the wrapper.
    try:
        from comfy_kitchen.backends.eager.quantization import int8_linear as _orig
        _orig_int8_linear = _orig
    except Exception as e:
        print(f"{TAG} could not import eager int8_linear: {e!r}")
        return

    # Wrap the mixed_precision_ops factory so each generated int8 Linear routes
    # through the kernel W8A8 path. pick_operations() calls this by module name at
    # model load, so wrapping the module attribute is sufficient.
    try:
        import comfy.ops as ops

        orig_factory = ops.mixed_precision_ops
        if getattr(orig_factory, "_asfp8_wrapped", False):
            _installed = True
            return

        def wrapped_factory(*a, **k):
            cls = orig_factory(*a, **k)
            linear_cls = getattr(cls, "Linear", None)
            if linear_cls is not None and not getattr(linear_cls, "_asfp8_int8_patched", False):
                orig_forward = linear_cls.forward

                def forward(self, input, *args, **kwargs):
                    res = _try_int8_kernel_forward(self, input)
                    if res is not None:
                        return res
                    return orig_forward(self, input, *args, **kwargs)

                linear_cls.forward = forward
                linear_cls._asfp8_int8_patched = True
            return cls

        wrapped_factory._asfp8_wrapped = True
        ops.mixed_precision_ops = wrapped_factory
    except Exception as e:
        print(f"{TAG} could not wrap mixed_precision_ops: {e!r}")
        return

    _installed = True
    print(
        f"{TAG} INT8 convrot Linear routed through bit-exact Metal kernel on MPS "
        f"(clean W8A8: rotate->per-row quant->int8 matmul; weight-only fp32 "
        f"dequant/un-rotation bypassed)."
    )
