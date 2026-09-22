"""Patch: route comfy_kitchen INT8 convrot layers through our bit-exact Metal kernel.

Feeds int8_linear the raw bf16 activation so convrot rotates before the per-row quant,
skipping both comfy's lossy tensorwise pre-quant and the per-step fp32 weight dequant
and un-rotation. DEFAULT ON, gated on M5/Metal-4.1 + ninja; ASFP8_INT8_EXT=off disables.
"""

import sys

import torch

TAG = "[AppleSilicon-FP8/int8_kernel]"

_installed = False
_orig_int8_linear = None
_kernel = None
_kernel_tried = False
_self_checked = False
_self_ok = False

# gelu-erf is absent everywhere (kernel store, pybind, here): Metal has no `erf`
_ACT = {"none": 0, "silu": 1, "gelu_tanh": 2}


def _act_code(act):
    if act not in _ACT:
        raise ValueError(f"unknown act {act!r}; expected one of {sorted(_ACT)}")
    return _ACT[act]


def _apply_act(result, act):
    """Apply the named activation to an already-computed linear output."""
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
    mod = loader.module()
    if mod is not None:
        try:
            # warmup() dispatches for real, so a library that built but whose Metal
            # source the runtime rejects fails here
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
        print(f"{TAG} INT8 Metal kernel unavailable; using comfy's int8 path.")
    return _kernel


def _self_check():
    """One-time correctness gate on a tiny int8 matmul.

    The .so loading proves nothing: the Metal library is compiled on the first
    dispatch, so a toolchain that rejects it fails here. Call only after a layer has
    passed every eligibility gate.
    """
    global _self_checked, _self_ok
    if _self_checked:
        return _self_ok
    _self_checked = True
    try:
        g = torch.Generator().manual_seed(0)
        a = torch.randint(-127, 128, (32, 128), generator=g, dtype=torch.int8)
        w = torch.randint(-127, 128, (64, 128), generator=g, dtype=torch.int8)
        ref = a.int() @ w.int().t()
        out = _kernel.i8_matmul2d_nt(a.to("mps").contiguous(), w.to("mps").contiguous())
        _self_ok = torch.equal(out.cpu(), ref)
        if not _self_ok:
            print(f"{TAG} self-check mismatch; using comfy's int8 path.")
    except Exception as e:
        _self_ok = False
        print(f"{TAG} self-check raised; using comfy's int8 path: {e!r}")
    return _self_ok


def _verify():
    """The contract _caps.kernel_ready expects: build, warmup, then numerics."""
    return _ensure_kernel() is not None and _self_check()


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

    Semantics match the original exactly; only the matmul backend differs, and the
    weight is used in its stored [N,K] layout.
    """
    code = _act_code(act)  # raise on a typo before any dispatch

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

    # fused fast path: fold float(C)*row_scale[m] (+bias) into the kernel's store
    # epilogue, which is bit-identical to the chunked path below
    if (
        out_dtype == torch.bfloat16
        and weight_scale.numel() == 1
        and hasattr(_kernel, "i8_matmul2d_nt_fused")
    ):
        row_scale = (weight_scale.float() * x_scale.reshape(-1).float()).contiguous()
        bias_arg = bias.to(torch.bfloat16) if bias is not None else None
        # the activation is fused in-kernel here; do NOT also call _apply_act
        result = _kernel.i8_matmul2d_nt_fused(
            x_8.contiguous(), weight.contiguous(), row_scale, bias_arg, code
        )
        return result.reshape(*orig_shape[:-1], weight.shape[0])

    # C[M,N] int32 = x_8[M,K] @ weight[N,K]^T  (NT: weight in stored layout)
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

    result = _apply_act(result, act)

    return result.reshape(*orig_shape[:-1], weight.shape[0])


def _int8_swiglu_kernel(x, w_gate, w_up, ws_gate, ws_up, bias_gate=None, bias_up=None,
                        convrot=False, convrot_groupsize=256, act="silu"):
    """Fused gated linear: H = act(x@w_gate^T + bias_gate) * (x@w_up^T + bias_up).

    The single-pass kernel needs scalar weight scales; per-channel scales route through
    _int8_linear_kernel per branch. act in {"silu","gelu_tanh"}, never "none".
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

    The kernel build is attempted only after every cheap eligibility gate passes, so a
    model with no int8 convrot layers never pays for it.
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
        # offloaded weight: transient, so let comfy's forward cast it in rather than
        # latch mark_kernel_failed for the session
        if w.device.type != "mps":
            return None
        if getattr(self, "_full_precision_mm", False):
            return None
        # deliberately no comfy_force_cast_weights check: on a quantized weight that
        # cast IS the per-call dequant/un-rotation this path exists to bypass
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return None

        params = w._params
        if getattr(params, "transposed", False):
            return None

        from . import _caps
        if not _caps.kernel_ready("int8", _verify):
            return None

        qdata = w._qdata
        scale = params.scale
        convrot = getattr(params, "convrot", False)
        gs = getattr(params, "convrot_groupsize", 256)
        bias = self.bias

        return _int8_linear_kernel(input, qdata, scale, bias, input.dtype, convrot, gs)
    except Exception as e:
        # latch off: verification already passed, so this repeats on every later layer
        from . import _caps
        if _caps._kernel_ready.get("int8"):
            print(f"{TAG} kernel forward failed ({e!r}); using comfy's int8 path "
                  f"for the rest of this session.")
        _caps.mark_kernel_failed("int8")
        return None


_DEQUANT_MODE = None


def _dequant_enabled():
    """Opt-in gate for once-at-load weight dequant, plus a >= 48 GiB total-RAM check.

    Opt-in because the plain copy lands after comfy's model_management has budgeted
    around the loaded int8 size.
    """
    global _DEQUANT_MODE
    if _DEQUANT_MODE is None:
        import os
        v = os.environ.get("ASFP8_INT8_DEQUANT", "").strip().lower()
        _DEQUANT_MODE = False
        if v in ("1", "on", "true"):
            try:
                import psutil
                _DEQUANT_MODE = psutil.virtual_memory().total >= 48 * (1 << 30)
                if not _DEQUANT_MODE:
                    print(f"{TAG} ASFP8_INT8_DEQUANT=1 ignored: needs >= 48 GiB RAM "
                          f"for the dequantised weight copy.")
            except Exception:
                _DEQUANT_MODE = False
    return _DEQUANT_MODE


def _maybe_dequant_weight(self, input):
    """One-off per layer: replace an eligible int8 (or fp16) weight with a plain
    tensor in the compute dtype, resident on MPS.

    At M=1..2 the per-call rotate/quant/rescale dispatch overhead dominates the matmul,
    so paying the dequant once is faster and numerically identical to comfy's W8A16.
    """
    if getattr(self, "_asfp8_deq_done", False):
        return
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        # checks before the flag are properties of this call, so leave it unset for a
        # later one; everything past it is a stable property of the layer
        if not isinstance(input, torch.Tensor) or isinstance(input, QuantizedTensor):
            return
        if input.device.type != "mps":
            return
        # half precision only: fp32 would double the copy the RAM gate budgets for
        if input.dtype not in (torch.float16, torch.bfloat16):
            return
        # offloaded weight: dequantising would pull a bigger copy back onto the GPU
        # and undo the offload
        w = getattr(self, "weight", None)
        if w is None or getattr(w, "device", None) is None or w.device.type != "mps":
            return
        self._asfp8_deq_done = True
        if getattr(self, "_full_precision_mm", False):
            return
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return

        if isinstance(w, QuantizedTensor):
            if getattr(w, "_layout_cls", None) != "TensorWiseINT8Layout":
                return
            if getattr(w._params, "transposed", False):
                return
            plain = w.dequantize()
        elif isinstance(w, torch.Tensor) and w.dtype == torch.float16 and w.dtype != input.dtype:
            plain = w.detach()
        else:
            return
        if plain.dtype != input.dtype:
            plain = plain.to(input.dtype)
        if plain.device.type != "mps":
            plain = plain.to("mps")
        self.weight = torch.nn.Parameter(plain.contiguous(), requires_grad=False)
        b = getattr(self, "bias", None)
        if isinstance(b, torch.Tensor) and not isinstance(b, QuantizedTensor) and b.dtype != input.dtype:
            self.bias = torch.nn.Parameter(b.detach().to(device="mps", dtype=input.dtype), requires_grad=False)
        self.comfy_force_cast_weights = False
    except Exception as e:
        # latch on failure too, or the same error prints on every call
        self._asfp8_deq_done = True
        print(f"{TAG} weight dequant skipped ({e!r})")


def _maybe_dequant_embedding(self, dtype_hint):
    """One-off per Embedding: replace an int8 (or fp16) table with a plain tensor in
    the compute dtype, so cast_bias_weight stops casting it on every lookup."""
    if getattr(self, "_asfp8_deq_done", False):
        return
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        w = self.weight
        # off-MPS is transient (offload), so leave the flag unset for a later call
        if w is None or w.device.type != "mps":
            return
        self._asfp8_deq_done = True
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return
        # half precision only, same memory budget as _maybe_dequant_weight
        if dtype_hint not in (torch.float16, torch.bfloat16):
            dtype_hint = None
        target = dtype_hint if dtype_hint is not None else torch.bfloat16
        if isinstance(w, QuantizedTensor):
            plain = w.dequantize().to(target)
        elif isinstance(w, torch.Tensor) and w.dtype == torch.float16 and w.dtype != target:
            plain = w.detach().to(target)
        else:
            return
        self.weight = torch.nn.Parameter(plain.contiguous(), requires_grad=False)
        self.comfy_force_cast_weights = False
    except Exception as e:
        self._asfp8_deq_done = True
        print(f"{TAG} embedding dequant skipped ({e!r})")


def install():
    global _installed, _orig_int8_linear
    if _installed:
        return
    if sys.platform != "darwin":
        return
    from . import _caps
    if not _caps.resolve("ASFP8_INT8_EXT", default_on=True, cap=_caps.kernel_gate):
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    try:
        from comfy_kitchen.backends.eager.quantization import int8_linear as _orig
        _orig_int8_linear = _orig
    except Exception as e:
        print(f"{TAG} could not import eager int8_linear: {e!r}")
        return

    # pick_operations() calls the factory by module name at model load, so wrapping
    # the module attribute is enough
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
                    if _dequant_enabled():
                        _maybe_dequant_weight(self, input)
                    res = _try_int8_kernel_forward(self, input)
                    if res is not None:
                        return res
                    return orig_forward(self, input, *args, **kwargs)

                linear_cls.forward = forward
                linear_cls._asfp8_int8_patched = True
            emb_cls = getattr(cls, "Embedding", None)
            if emb_cls is not None and not getattr(emb_cls, "_asfp8_int8_patched", False):
                orig_emb_forward = emb_cls.forward

                def emb_forward(self, input, *args, **kwargs):
                    if _dequant_enabled():
                        hint = kwargs.get("out_dtype", args[0] if args else None)
                        _maybe_dequant_embedding(self, hint)
                    return orig_emb_forward(self, input, *args, **kwargs)

                emb_cls.forward = emb_forward
                emb_cls._asfp8_int8_patched = True
            return cls

        wrapped_factory._asfp8_wrapped = True
        ops.mixed_precision_ops = wrapped_factory

        # ops.linear_input_act (fused swiglu) reads linear.weight directly and never
        # calls Linear.forward, so the dequant has to hook here too. getattr: a comfy
        # without it must not abort install() after the wrap above.
        orig_lia = getattr(ops, "linear_input_act", None)
        if orig_lia is not None and not getattr(orig_lia, "_asfp8_deq_wrapped", False):
            def linear_input_act(linear, x, input_act, *args, **kwargs):
                if _dequant_enabled():
                    _maybe_dequant_weight(linear, x)
                return orig_lia(linear, x, input_act, *args, **kwargs)

            linear_input_act._asfp8_deq_wrapped = True
            ops.linear_input_act = linear_input_act
    except Exception as e:
        print(f"{TAG} could not wrap mixed_precision_ops: {e!r}")
        return

    _installed = True
    deq = "on" if _dequant_enabled() else "off"
    print(
        f"{TAG} INT8 convrot Linear seam installed on MPS (clean W8A8: "
        f"rotate->per-row quant->int8 matmul; weight-only fp32 dequant/un-rotation "
        f"bypassed). The kernel builds and runs its bit-exact self-check on the "
        f"first int8 layer, and only routes if that passes. "
        f"Once-at-load weight dequant: {deq} (opt-in: ASFP8_INT8_DEQUANT=1, "
        f"needs >= 48 GiB RAM)."
    )
