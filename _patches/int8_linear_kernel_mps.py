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
_self_checked = False
_self_ok = False

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
    mod = loader.module()
    if mod is not None:
        try:
            # warmup() dispatches for real, so it is where a library that BUILT
            # but whose Metal source the runtime rejects actually fails (#13).
            mod.warmup()
        except Exception as e:
            print(f"{TAG} warmup failed; disabling: {e!r}")
            return None
    return mod


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


def _self_check():
    """One-time correctness gate on a tiny int8 matmul.

    The .so loading is not proof the kernel works: the Metal library is compiled
    by newLibraryWithSource on the FIRST dispatch, so a toolchain that rejects it
    fails here rather than at build time. Without this latch that compile is
    retried on every eligible Linear (issue #13) — slower than not having the
    kernel at all. Mirrors fp8_linear_kernel_mps._self_check.

    MUST be called only AFTER a layer passes every eligibility gate.
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
    """The contract _caps.kernel_ready expects: build, warmup, then numerics.

    Nothing short of this proves the kernel works. kernel_gate() only checks that
    the chip has matrix units and that ninja exists; it says nothing about whether
    int8_gemm.mm builds here or what it computes (#14, #25, #27).
    """
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
        # Offloaded weight (comfy parks it on CPU between uses): fall back to the
        # comfy forward, which casts it in. Reaching _int8_linear_kernel with a
        # CPU weight would throw in its own fallback and latch mark_kernel_failed,
        # killing the kernel for the session over a transient condition.
        if w.device.type != "mps":
            return None
        if getattr(self, "_full_precision_mm", False):
            return None
        # comfy_force_cast_weights on an int8 QuantizedTensor only means "storage
        # dtype != compute dtype, cast at use" -- and for a quantized weight that
        # cast IS the per-call W8A16 dequant/un-rotation this path exists to
        # bypass, so it must not disqualify the kernel route. (MiniMax Music 3's
        # text encoder sets it on every layer, which silently forced the slow
        # path for the whole model.)
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
        # Never take down a render; fall back to comfy's original forward, and
        # stop using the kernel for the rest of the session -- verification
        # already passed, so this repeats on every later layer if we don't.
        from . import _caps
        if _caps._kernel_ready.get("int8"):
            print(f"{TAG} kernel forward failed ({e!r}); using comfy's int8 path "
                  f"for the rest of this session.")
        _caps.mark_kernel_failed("int8")
        return None


_DEQUANT_MODE = None


def _dequant_enabled():
    """Opt-in gate for once-at-load weight dequant: ASFP8_INT8_DEQUANT=1 enables,
    subject to a >= 48 GiB total-RAM safety check; unset or off stays off.

    Opt-in because the plain copy (~16 GB for an 8B int8 model) lands AFTER
    comfy's model_management has budgeted around the loaded int8 size, so
    auto-enabling near the threshold risks thrash/OOM comfy can't see coming."""
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
    """One-off per layer: replace an eligible int8 QuantizedTensor weight with its
    dequantised (un-rotated) plain tensor in the compute dtype, resident on MPS.

    Rationale: single-token AR decode is memory-bandwidth-bound, and every
    quantised route (kernel W8A8 or comfy's per-call W8A16 dequant) pays a
    per-call rotate/quant/rescale op chain whose MPS dispatch overhead dominates
    at M=1..2 (measured 3-10x the matmul cost on MiniMax Music 3 AR decode).
    Paying the dequant once and running single-dispatch F.linear is both faster
    and numerically identical to comfy's stock W8A16 path. fp16 weights with a
    different compute dtype get the same treatment: manual_cast otherwise copies
    the full tensor every call (MiniMax's 134 MB audio_heads dominate a decode
    profile through aten::copy_)."""
    if getattr(self, "_asfp8_deq_done", False):
        return
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        # Transient conditions (properties of this call's input) leave the flag
        # unset so a later MPS-resident call can still dequantise. Everything
        # past the flag is a stable property of the layer, decided once.
        if not isinstance(input, torch.Tensor) or isinstance(input, QuantizedTensor):
            return
        if input.device.type != "mps":
            return
        # Half-precision compute only: an fp32 activation would balloon the
        # dequantised copy to 2x the size the >= 48 GiB gate budgets for.
        if input.dtype not in (torch.float16, torch.bfloat16):
            return
        self._asfp8_deq_done = True
        if getattr(self, "_full_precision_mm", False):
            return
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return

        w = self.weight
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
        # Latch on failure too: retrying an identical dequant every call would
        # just repeat the error (and the print) for the rest of the session.
        self._asfp8_deq_done = True
        print(f"{TAG} weight dequant skipped ({e!r})")


def _maybe_dequant_embedding(self, dtype_hint):
    """One-off per Embedding: replace an int8 QuantizedTensor (or fp16) table with
    a plain tensor in the compute dtype so per-call cast/dequant of the whole
    table disappears (cast_bias_weight casts the full weight every lookup)."""
    if getattr(self, "_asfp8_deq_done", False):
        return
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        w = self.weight
        # Off-MPS is transient (offload); leave the flag unset for a later call.
        if w is None or w.device.type != "mps":
            return
        self._asfp8_deq_done = True
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return
        # The hint comes from Embedding.forward's out_dtype (kwarg or first
        # positional); anything that isn't a half-precision dtype is ignored,
        # not trusted (same memory-budget reasoning as _maybe_dequant_weight).
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
    # DEFAULT ON, gated on M5-class matrix units + ninja to build the ObjC++
    # extension). On unsupported HW the default resolves to OFF so we never attempt a
    # build; ASFP8_INT8_EXT=off force-disables, =1 forces the build attempt anyway.
    from . import _caps
    if not _caps.resolve("ASFP8_INT8_EXT", default_on=True, cap=_caps.kernel_gate):
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

        # down_proj is invoked via ops.linear_input_act (fused swiglu path), which
        # reads linear.weight directly and never calls Linear.forward -- so the
        # once-at-load dequant must hook here too or those layers stay int8.
        # getattr, not attribute access: a comfy without linear_input_act must not
        # abort install() here -- mixed_precision_ops is already wrapped above and
        # the except below would leave _installed False with a misleading message.
        orig_lia = getattr(ops, "linear_input_act", None)
        if orig_lia is not None and not getattr(orig_lia, "_asfp8_deq_wrapped", False):
            def linear_input_act(linear, x, input_act):
                if _dequant_enabled():
                    _maybe_dequant_weight(linear, x)
                return orig_lia(linear, x, input_act)

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
