"""Fix: torch._scaled_mm with FP8 inputs on MPS (FLUX, SD3.5, etc.).

MPS has no fp8 kernel for it, so decode both operands to bf16 through the LUT, run the
matmul on the matrix units, then apply scales and bias in fp32. Where the fp8-native
Metal kernel is available (DEFAULT ON, ASFP8_FP8_EXT=off disables) large e4m3 matmuls
skip the decode instead.
"""

import os

import torch

from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/scaled_mm]"

_original = None
_original_v2 = None
_installed = False

# torch >= 2.11 only; resolved once, since the v2 predicates sit on the hot path
try:
    from torch.nn.functional import ScalingType as _ScalingType
except Exception:
    _ScalingType = None
try:
    from torch.nn.functional import SwizzleType as _SwizzleType
except Exception:
    _SwizzleType = None

_backend = None          # the cpp module, or False once known-unavailable
_self_checked = False


def _compute_dtype(out_dtype):
    """bf16 for bf16/fp16 results (matrix units, and fp32 range so no overflow);
    float32 only when the caller explicitly asks for f32."""
    if out_dtype in (torch.bfloat16, torch.float16):
        return torch.bfloat16
    return torch.float32


def _decode(t, compute_dtype):
    if t.dtype in FP8_DTYPES:
        return decode_fp8(t, compute_dtype)
    return t.to(compute_dtype)


def _min_dim():
    """Weight-size threshold below which the fp8-native kernel costs more than it saves."""
    try:
        return int(os.environ.get("ASFP8_FP8_EXT_MIN_DIM", "8192"))
    except ValueError:
        return 8192


def _fast_eligible(input, other):
    """Shape/dtype/capability predicate; no per-call device work, probes are memoised."""
    from . import _caps
    if not _caps.resolve("ASFP8_FP8_EXT", default_on=True, cap=_caps.kernel_gate):
        return False
    # the kernel decodes both operands as e4m3, so only route when that is exact
    if input.dtype != torch.float8_e4m3fn or other.dtype != torch.float8_e4m3fn:
        return False
    if input.device.type != "mps" or other.device.type != "mps":
        return False
    if input.dim() != 2 or other.dim() != 2:
        return False
    # input [M,K] @ other [K,N]; weight recovered as other.t() = [N,K]
    K = int(input.shape[1])
    N = int(other.shape[1])
    return max(N, K) >= _min_dim()


def _get_backend():
    """Lazily build + parity-self-check the fp8 extension. Returns the module or None
    (permanently, after a failure) so the caller falls back to the decode path."""
    global _backend, _self_checked
    if _backend is not None:
        return _backend or None
    from .fp8_ext.loader import module
    mod = module()
    if mod is None:
        _backend = False
        return None
    if not _self_checked:
        _self_checked = True
        try:
            # local generator, never torch.manual_seed: this runs mid-render, so
            # reseeding the process RNG would land inside a user's sampling
            g = torch.Generator().manual_seed(0)
            a = (torch.randn(64, 8192, generator=g) * 0.3).to(torch.float8_e4m3fn)
            w = (torch.randn(8192, 8192, generator=g) * 0.3).to(torch.float8_e4m3fn)   # [N,K]
            a_u8 = a.view(torch.uint8).to("mps").contiguous()
            w_u8 = w.view(torch.uint8).to("mps").contiguous()
            ref = decode_fp8(a.to("mps"), torch.float32) @ decode_fp8(w.to("mps"), torch.float32).t()
            raw = mod.fp8fp8_matmul2d_nt(a_u8, w_u8, 8192, 8192)
            rel = ((raw - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
            if rel >= 5e-2:
                print(f"{TAG} fp8-native self-check failed (rel={rel:.4f}); using decode path.")
                _backend = False
                return None
        except Exception as e:
            print(f"{TAG} fp8-native self-check raised; using decode path: {e!r}")
            _backend = False
            return None
    _backend = mod
    print(f"{TAG} fp8-native scaled_mm enabled (Metal 4.1 ext; min_dim={_min_dim()}).")
    return _backend


def _fast_route(input, other, scale_a, scale_b, scale_result, bias, out_dtype):
    """fp8xfp8 -> f32 via the Metal 4.1 kernel, then scales/bias in f32. Raises on any
    failure so the caller delegates to the decode implementation."""
    mod = _get_backend()
    if mod is None:
        raise RuntimeError("fp8 extension unavailable")
    K = int(input.shape[1])
    N = int(other.shape[1])
    a_u8 = input.contiguous().view(torch.uint8)
    # other is [K,N]; the kernel wants W=[N,K] contiguous fp8 bytes
    w_u8 = other.t().contiguous().view(torch.uint8)
    out = mod.fp8fp8_matmul2d_nt(a_u8, w_u8, K, N)   # [M,N] f32, unscaled
    if scale_a is not None:
        out = out * scale_a.to(torch.float32)
    if scale_b is not None:
        out = out * scale_b.to(torch.float32)
    if scale_result is not None:
        out = out * scale_result.to(torch.float32)
    if bias is not None:
        out = out + bias.to(torch.float32)
    if out_dtype is not None:
        out = out.to(out_dtype)
    return out


def _mps_scaled_mm(
    input,
    other,
    *,
    out_dtype=None,
    scale_a=None,
    scale_b=None,
    bias=None,
    scale_result=None,
    use_fast_accum=False,
):
    is_mps = input.device.type == "mps"
    is_fp8 = input.dtype in FP8_DTYPES or other.dtype in FP8_DTYPES
    if not (is_mps and is_fp8):
        return _original(
            input, other,
            out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b,
            bias=bias, scale_result=scale_result, use_fast_accum=use_fast_accum,
        )

    # any failure here falls through to the decode path below, so a render never breaks
    if _fast_eligible(input, other):
        try:
            return _fast_route(input, other, scale_a, scale_b, scale_result, bias, out_dtype)
        except Exception as e:
            print(f"{TAG} fp8-native path failed ({e!r}); delegating to decode path.")

    compute_dtype = _compute_dtype(out_dtype)

    # input: (M,K), other: (K,N) column-major — torch._scaled_mm's layout
    a = _decode(input, compute_dtype)
    b = _decode(other, compute_dtype)

    out = a @ b

    # the scales factor out of the dot product, so apply them to the result; in fp32
    # to avoid intermediate rounding
    if scale_a is not None or scale_b is not None or scale_result is not None:
        acc = out.to(torch.float32)
        if scale_a is not None:
            acc = acc * scale_a.to(torch.float32)
        if scale_b is not None:
            acc = acc * scale_b.to(torch.float32)
        if scale_result is not None:
            acc = acc * scale_result.to(torch.float32)
        out = acc.to(out.dtype)

    if bias is not None:
        # always in f32, so bf16 compute doesn't lose the bias term before the widen
        out = out.to(torch.float32) + bias.to(torch.float32)
    if out_dtype is not None:
        out = out.to(out_dtype)
    return out


def _is_tensorwise(recipe):
    """True only for the plain TensorWise recipe; NVFP4/MXFP8 pass a list-valued pair."""
    if _ScalingType is None or isinstance(recipe, (list, tuple)):
        return False
    return recipe == _ScalingType.TensorWise


def _no_swizzle(swizzle):
    if swizzle is None:
        return True
    if _SwizzleType is None or isinstance(swizzle, (list, tuple)):
        return False
    return swizzle == _SwizzleType.NO_SWIZZLE


def _mps_scaled_mm_v2(
    mat_a,
    mat_b,
    scale_a,
    scale_recipe_a,
    scale_b,
    scale_recipe_b,
    swizzle_a=None,
    swizzle_b=None,
    bias=None,
    output_dtype=torch.bfloat16,
    contraction_dim=(),
    use_fast_accum=False,
):
    """Wrapper for torch.nn.functional.scaled_mm — the `aten::_scaled_mm_v2` seam.

    comfy_kitchen prefers this API on a bare hasattr, so without this wrapper the
    seam below goes dark. Plain TensorWise fp8 on MPS delegates to the legacy path;
    microscaling recipes, list-valued scales and contraction_dim fall through.
    """
    is_mps = isinstance(mat_a, torch.Tensor) and mat_a.device.type == "mps"
    is_fp8 = (
        isinstance(mat_a, torch.Tensor) and mat_a.dtype in FP8_DTYPES
    ) or (
        isinstance(mat_b, torch.Tensor) and mat_b.dtype in FP8_DTYPES
    )
    plain_scales = isinstance(scale_a, torch.Tensor) and isinstance(scale_b, torch.Tensor)

    if (
        is_mps
        and is_fp8
        and plain_scales
        and _is_tensorwise(scale_recipe_a)
        and _is_tensorwise(scale_recipe_b)
        and _no_swizzle(swizzle_a)
        and _no_swizzle(swizzle_b)
        and not contraction_dim
    ):
        return _mps_scaled_mm(
            mat_a,
            mat_b,
            out_dtype=output_dtype,
            scale_a=scale_a,
            scale_b=scale_b,
            bias=bias,
        )

    if _original_v2 is None:
        # only reachable if bound onto F.scaled_mm without going through install()
        raise RuntimeError(
            f"{TAG} F.scaled_mm wrapper is installed but has no original to "
            "delegate to — install() did not complete."
        )

    return _original_v2(
        mat_a,
        mat_b,
        scale_a,
        scale_recipe_a,
        scale_b,
        scale_recipe_b,
        swizzle_a=swizzle_a,
        swizzle_b=swizzle_b,
        bias=bias,
        output_dtype=output_dtype,
        contraction_dim=contraction_dim,
        use_fast_accum=use_fast_accum,
    )


def install():
    global _original, _original_v2, _installed
    if _installed:
        return
    if not hasattr(torch, "_scaled_mm"):
        return  # requires PyTorch 2.4+
    _original = torch._scaled_mm
    torch._scaled_mm = _mps_scaled_mm
    msg = f"{TAG} torch._scaled_mm FP8 on MPS via LUT decode + bf16 matrix-unit matmul."
    # comfy_kitchen prefers F.scaled_mm wherever it exists, so the seam above goes
    # dark unless this one is wrapped too
    if hasattr(torch.nn.functional, "scaled_mm"):
        _original_v2 = torch.nn.functional.scaled_mm
        torch.nn.functional.scaled_mm = _mps_scaled_mm_v2
        msg += " F.scaled_mm (v2 seam) wrapped too."
    _installed = True
    print(msg)
