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

DEFAULT ON, gated on M5/Metal-4.1 + ninja; ``ASFP8_INT8_EXT=off`` force-disables, ``=1`` forces the build attempt.
"""

import sys

import torch

TAG = "[AppleSilicon-FP8/int8_kernel]"

_installed = False
_orig_int8_linear = None
_kernel = None
_kernel_tried = False

# Supported fused activations. P0 verdict (M5 Max / macOS 27 / Metal 4.1):
# Metal `erf` is unavailable, so act=3 (gelu-erf) is dropped entirely; only
# {none, silu, gelu_tanh} are kept everywhere (kernel store, pybind, this map).
_ACT = {"none": 0, "silu": 1, "gelu_tanh": 2}


def _act_code(act):
    if act not in _ACT:
        raise ValueError(f"unknown act {act!r}; expected one of {sorted(_ACT)}")
    return _ACT[act]


def _apply_act(result, act):
    """Apply the named activation to an already-computed linear output. Single source
    of truth so no return path can silently drop the activation."""
    if act == "none":
        return result
    if act == "silu":
        return torch.nn.functional.silu(result)
    if act == "gelu_tanh":
        return torch.nn.functional.gelu(result, approximate="tanh")
    raise ValueError(f"unknown act {act!r}; expected one of {sorted(_ACT)}")


def _load_kernel():
    try:
        from .int8_ext import loader
    except Exception as e:  # pragma: no cover - import wiring
        print(f"{TAG} loader import failed: {e!r}")
        return None
    return loader.module()


def _ensure_kernel():
    """Build the Metal extension lazily, on the FIRST layer that can actually use it.

    Building inside install() blocked ComfyUI's startup import on a synchronous
    ninja+clang build (issue: "hangs forever at startup when ninja is installed").
    Deferring it here matches scaled_mm_fp8._get_backend / int4_linear_mps._load_kernel
    and keeps startup non-blocking; a failed build is remembered so we retry once only.
    """
    global _kernel, _kernel_tried
    if _kernel is not None or _kernel_tried:
        return _kernel
    _kernel_tried = True
    _kernel = _load_kernel()
    if _kernel is None:
        print(f"{TAG} INT8 Metal kernel unavailable; using comfy's int8 path.")
    return _kernel


def _int8_linear_kernel(
    x,
    weight,
    weight_scale,
    bias=None,
    out_dtype=torch.bfloat16,
    convrot=False,
    convrot_groupsize=256,
    act="none",
):
    """Kernel-backed drop-in for comfy_kitchen eager int8_linear.

    Semantics match the original exactly (optional convrot activation rotation,
    per-row int8 activation quant, int8xint8->int32 matmul, chunked rescale by
    weight_scale*row_scale, optional bias). Only the matmul backend differs (our
    NT kernel vs torch._int_mm) and the weight is used in its stored [N,K] layout.
    """
    code = _act_code(act)  # validate up front (raises on typo) before any dispatch

    if (
        _kernel is None
        or x.device.type != "mps"
        or weight.device.type != "mps"
        or weight.dtype != torch.int8
    ):
        if _orig_int8_linear is None:
            raise RuntimeError(
                f"{TAG} _int8_linear_kernel fallback needs the original int8_linear, "
                "but it is None (install() never ran / kernel not built). Call install() "
                "first or route through the comfy forward, which catches this."
            )
        return _apply_act(
            _orig_int8_linear(
                x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize
            ),
            act,
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
        # Activation is fused IN-KERNEL here; do NOT also call _apply_act.
        result = _kernel.i8_matmul2d_nt_fused(
            x_8.contiguous(), weight.contiguous(), row_scale, bias_arg, code
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

    # Chunked fallback path: activation is applied here (never fused in-kernel).
    result = _apply_act(result, act)

    return result.reshape(*orig_shape[:-1], weight.shape[0])


def _int8_swiglu_kernel(x, w_gate, w_up, ws_gate, ws_up, bias_gate=None, bias_up=None,
                        convrot=False, convrot_groupsize=256, act="silu"):
    """Fused gated linear: H = act(x@w_gate^T + bias_gate) * (x@w_up^T + bias_up).

    Uses the single-pass fused gate kernel only when both weight scales are scalar
    (the row-scale precombine ws*x_scale[m] is only valid then); otherwise routes
    through the per-branch _int8_linear_kernel path (which handles per-channel scales
    and applies the activation itself). act in {"silu","gelu_tanh"} (no "none").
    """
    if act == "none":
        raise ValueError("_int8_swiglu_kernel requires a real gate activation, not 'none'")
    _act_code(act)  # validate (raises on typo)
    if w_gate.shape != w_up.shape:
        raise ValueError(
            f"gate/up weights must match: {tuple(w_gate.shape)} vs {tuple(w_up.shape)}"
        )

    scalar_scales = (ws_gate.numel() == 1 and ws_up.numel() == 1)
    fused_ok = (_kernel is not None and x.device.type == "mps" and w_gate.dtype == torch.int8
                and w_up.dtype == torch.int8 and scalar_scales
                and hasattr(_kernel, "i8_matmul2d_nt_swiglu"))
    if not fused_ok:
        # Per-branch fallback (kernel missing, off-MPS, non-int8, or per-channel scales —
        # the row-scale precombine below is only valid for scalar weight scales).
        g = _int8_linear_kernel(x, w_gate, ws_gate, bias_gate, torch.bfloat16,
                                convrot, convrot_groupsize, act=act)  # activation applied here
        u = _int8_linear_kernel(x, w_up, ws_up, bias_up, torch.bfloat16,
                                convrot, convrot_groupsize, act="none")
        return g * u

    from comfy_kitchen.backends.eager.quantization import quantize_int8_rowwise
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
    if convrot:
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x = _rotate_activation(x, h, convrot_groupsize)
    shp = x.shape
    x2 = x.reshape(-1, x.shape[-1])
    x8, xs = quantize_int8_rowwise(x2)
    xs = xs.reshape(-1).float()
    rsg = (ws_gate.view(-1).float() * xs).contiguous()  # scalar * [M] -> [M] (valid: scalar_scales)
    rsu = (ws_up.view(-1).float() * xs).contiguous()
    bg = bias_gate.to(torch.bfloat16) if bias_gate is not None else None
    bu = bias_up.to(torch.bfloat16) if bias_up is not None else None
    out = _kernel.i8_matmul2d_nt_swiglu(x8.contiguous(), w_gate.contiguous(),
            w_up.contiguous(), rsg, rsu, bg, bu, _act_code(act))
    return out.reshape(*shp[:-1], w_gate.shape[0])


def _try_int8_kernel_forward(self, input):
    """Return the layer output via the kernel W8A8 path, or None to fall back.

    Eligible iff: input is a plain MPS Tensor (not a QuantizedTensor); weight is a
    TensorWiseINT8Layout QuantizedTensor; not full-precision-mm; no force-cast; no
    LoRA weight/bias functions; weight not logically transposed; and the Metal
    kernel builds. The build is attempted only AFTER every cheap eligibility gate
    passes, so a model with no int8 convrot layers never pays for it.
    """
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

        if _ensure_kernel() is None:
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
    global _installed, _orig_int8_linear
    if _installed:
        return
    if sys.platform != "darwin":
        return
    # DEFAULT ON, gated on Tier B (Metal-4 tensor ops + ninja to build the ObjC++
    # extension). On unsupported HW the default resolves to OFF so we never attempt a
    # build; ASFP8_INT8_EXT=off force-disables, =1 forces the build attempt anyway.
    from . import _caps
    if not _caps.resolve("ASFP8_INT8_EXT", default_on=True, cap=_caps.tier_b_ready):
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

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
        f"dequant/un-rotation bypassed). Kernel builds on first int8 layer, not now."
    )
