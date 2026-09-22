"""Fix: ConvRot W4A4 (int4) models crawl on MPS, slower per Linear than int8.

comfy-kitchen's eager emulation quantizes the activation to int4 and then unpacks both
operands back to float anyway, so skipping that step is both faster and more accurate
(W4A16, not fake W4A4). The stored weight is already rotated, so rotating only the
activation leaves x_rot @ W_rot^T == x @ W^T and no per-call un-rotation is needed.
Non-MPS inputs fall through; a no-op where comfy-kitchen has no ConvRot W4A4 layout.
"""

import sys

import torch

TAG = "[AppleSilicon-FP8/int4_linear]"

_installed = False
_patched = False
_orig = None
_kernel = None
_kernel_tried = False


def _load_kernel():
    """W4A8 Metal kernel (opt-in, ASFP8_INT4_EXT=1): int8 act x packed-int4 weight,
    with the per-row dequant and bias fused into the store epilogue."""
    global _kernel, _kernel_tried
    if _kernel_tried:
        return _kernel
    _kernel_tried = True
    try:
        from .int4_ext import loader
        _kernel = loader.module()
        if _kernel is not None:
            # compile now, so an unsupported GPU/SDK raises here rather than mid-render
            _kernel.warmup()
    except Exception as e:
        print(f"{TAG} int4 kernel unavailable ({e!r}); using W4A16 fast path.")
        _kernel = None
    if _kernel is not None:
        print(f"{TAG} W4A8 fused Metal kernel active for ConvRot int4 layers.")
    return _kernel


def _w4a8_kernel_linear(x_rot_2d, qweight, wscales, bias, kernel):
    absmax = x_rot_2d.float().abs().amax(dim=-1).clamp(min=1e-10)
    x_scale = absmax / 127.0
    qx = torch.round(x_rot_2d.float() / x_scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    b = bias.to(device=x_rot_2d.device, dtype=torch.bfloat16).contiguous() if bias is not None else None
    y = kernel.i8i4_linear_fused_nt(
        qx.contiguous(), qweight.contiguous(),
        x_scale.contiguous(), wscales.float().contiguous(), b,
        x_rot_2d.shape[-1], qweight.shape[0])
    return y if x_rot_2d.dtype == torch.bfloat16 else y.to(x_rot_2d.dtype)


def _unpack_int4_signed_fast(packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Decode row-major packed signed nibbles (low = even column) via arithmetic shifts."""
    lo = (packed << 4) >> 4
    hi = packed >> 4
    out = torch.stack((lo, hi), dim=-1).to(dtype)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


_self_checked = False
_self_ok = False


def _self_check():
    """One-time numeric gate on the real W4A8 kernel.

    warmup() only proves the shader compiles, not that the int4b nibble order still
    matches kitchen's packing or that the fused dequant epilogue is right.
    """
    global _self_checked, _self_ok
    if _self_checked:
        return _self_ok
    _self_checked = True
    try:
        M, N, K = 32, 64, 128
        g = torch.Generator().manual_seed(0)
        qx = torch.randint(-127, 128, (M, K), generator=g, dtype=torch.int8)
        packed = torch.randint(-128, 128, (N, K // 2), generator=g, dtype=torch.int8)
        x_scale = torch.rand(M, generator=g, dtype=torch.float32) * 0.01 + 0.001
        wscales = torch.rand(N, generator=g, dtype=torch.float32) * 0.01 + 0.001

        w = _unpack_int4_signed_fast(packed, torch.float32)
        ref = (qx.float() @ w.t()) * x_scale.unsqueeze(-1) * wscales.unsqueeze(0)

        out = _kernel.i8i4_linear_fused_nt(
            qx.to("mps").contiguous(), packed.to("mps").contiguous(),
            x_scale.to("mps").contiguous(), wscales.to("mps").contiguous(),
            None, K, N)
        # the epilogue stores bf16, so compare at bf16 precision, not exactly
        _self_ok = torch.allclose(out.cpu().float(), ref, rtol=3e-2, atol=3e-2)
        if not _self_ok:
            print(f"{TAG} int4 self-check mismatch; using the W4A16 path.")
    except Exception as e:
        _self_ok = False
        print(f"{TAG} int4 self-check raised; using the W4A16 path: {e!r}")
    return _self_ok


def _verify():
    """The contract _caps.kernel_ready expects: build + warmup, then numerics."""
    return _load_kernel() is not None and _self_check()


def _w4a16_linear_mps(x, qweight, wscales, bias, convrot_groupsize, mod):
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    h = mod._build_hadamard(convrot_groupsize, device=x2d.device, dtype=x2d.dtype)
    x_rot = mod._rotate_activation(x2d, h, convrot_groupsize)

    from . import _caps
    # deliberately no chip pre-filter, unlike int8/fp8: int4 is opt-in, so reaching a
    # build at all means the user asked for it explicitly, and the self-check below
    # still rejects a wrong result on pre-M5 hardware
    kernel = _load_kernel() if _caps.kernel_ready("int4", _verify) else None
    if kernel is not None:
        try:
            y = _w4a8_kernel_linear(x_rot, qweight, wscales, bias, kernel)
            return y.reshape(*orig_shape[:-1], qweight.shape[0])
        except Exception as e:
            # latch off: verification already passed, so this recurs on every later layer
            print(f"{TAG} W4A8 kernel call failed ({e!r}); using W4A16 for the "
                  f"rest of this session.")
            _caps.mark_kernel_failed("int4")

    w = _unpack_int4_signed_fast(qweight, x2d.dtype)
    w = w * wscales.to(device=w.device, dtype=x2d.dtype).reshape(-1, 1)
    b = bias.to(device=x2d.device, dtype=x2d.dtype) if bias is not None else None
    y = torch.nn.functional.linear(x_rot, w, b)
    return y.reshape(*orig_shape[:-1], qweight.shape[0])


def install():
    global _installed, _patched, _orig
    if _installed:
        return
    _installed = True

    if not (hasattr(torch, "mps") and torch.backends.mps.is_available()):
        return

    try:
        import comfy_kitchen.tensor.convrot_w4a4 as ck_convrot
        import comfy_kitchen.backends.eager.convrot_w4a4 as ck_eager
    except ImportError:
        print(f"{TAG} comfy-kitchen has no ConvRot W4A4 layout (too old) — skipping.")
        return

    _orig = ck_convrot.convrot_w4a4_linear

    def convrot_w4a4_linear_mps(x, qweight, wscales, bias=None, convrot_groupsize=256,
                                quant_group_size=64, linear_dtype="int4"):
        if x.device.type != "mps":
            return _orig(x, qweight, wscales, bias=bias, convrot_groupsize=convrot_groupsize,
                         quant_group_size=quant_group_size, linear_dtype=linear_dtype)
        return _w4a16_linear_mps(x, qweight, wscales, bias, convrot_groupsize, ck_eager)

    ck_convrot.convrot_w4a4_linear = convrot_w4a4_linear_mps
    _patched = True
    print(f"{TAG} ConvRot W4A4 (int4) Linear on MPS -> W4A16 rotated-basis fast path.")


def uninstall():
    global _patched
    if _patched and _orig is not None:
        sys.modules["comfy_kitchen.tensor.convrot_w4a4"].convrot_w4a4_linear = _orig
        _patched = False
