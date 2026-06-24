"""Patch #15 (EXPERIMENTAL, opt-in): route large fp8 MLP Linears through the Metal 4.1
fp8-native matmul2d extension on Apple Silicon. Default OFF (ASFP8_FP8_EXT=1 to enable).

Wraps F.linear *after* patch #10. On the eligible fast path (opt-in, MPS, fp8 weight,
large MLP shape, extension built + parity-checked) it computes x @ Wᵀ via the fp8 kernel
(reads 1-byte weights, no bf16 materialization, fp32 accumulate). Otherwise it delegates
to the previously-installed F.linear (#10's decode->bf16->MPS path). Any failure ->
permanent delegate. See docs/superpowers/specs/2026-06-24-fp8-native-linear-productionization-design.md.
"""

import os
import sys

import torch

TAG = "[AppleSilicon-FP8/fp8_linear]"

_orig = None
_installed = False
_backend = None          # the cpp module, or False once known-unavailable
_self_checked = False


def _min_dim():
    try:
        return int(os.environ.get("ASFP8_FP8_EXT_MIN_DIM", "8192"))
    except ValueError:
        return 8192


def _weight_eligible(weight):
    """Pure shape/dtype/env predicate (no device work)."""
    if os.environ.get("ASFP8_FP8_EXT") != "1":
        return False
    if getattr(weight, "dtype", None) != torch.float8_e4m3fn:
        return False
    if getattr(getattr(weight, "device", None), "type", None) != "mps":
        return False
    if weight.dim() != 2:
        return False
    out_f, in_f = weight.shape[0], weight.shape[1]
    return max(int(out_f), int(in_f)) >= _min_dim()


def _get_backend():
    """Lazily build + parity-self-check the extension. Returns the module or None
    (permanently, after a failure)."""
    global _backend, _self_checked
    if _backend is not None:
        return _backend or None
    from _patches.fp8_ext.loader import module
    mod = module()
    if mod is None:
        _backend = False
        return None
    if not _self_checked:
        _self_checked = True
        try:
            from _patches._common import decode_fp8
            torch.manual_seed(0)
            x = (torch.randn(64, 8192) * 0.3).to(torch.half).to("mps").contiguous()
            w = (torch.randn(8192, 8192) * 0.3).to(torch.float8_e4m3fn).contiguous()
            wu = w.view(torch.uint8).to("mps").contiguous()
            ref = x.float() @ decode_fp8(w.to("mps"), torch.float32).t()
            out = mod.fp8_matmul2d_nt(x, wu, 8192)
            rel = ((out - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
            if rel >= 5e-2:
                print(f"{TAG} self-check failed (rel={rel:.4f}); fp8-native disabled.")
                _backend = False
                return None
        except Exception as e:
            print(f"{TAG} self-check raised; fp8-native disabled: {e!r}")
            _backend = False
            return None
    _backend = mod
    print(f"{TAG} fp8-native Linear enabled (Metal 4.1 ext; min_dim={_min_dim()}).")
    return _backend


def _route(inp, weight, bias):
    """Eligible fast path: out = x @ Wᵀ + b via fp8 kernel. inp real dtype, weight fp8 [N,K]."""
    mod = _get_backend()
    N, K = int(weight.shape[0]), int(weight.shape[1])
    orig_shape = inp.shape
    x2d = inp.reshape(-1, K).to(torch.half).contiguous()
    w_u8 = weight.view(torch.uint8).contiguous()
    out = mod.fp8_matmul2d_nt(x2d, w_u8, N)           # [M, N] f32
    if bias is not None:
        out = out + bias.to(torch.float32)
    out = out.reshape(*orig_shape[:-1], N).to(inp.dtype)
    return out


def _linear(inp, weight, bias=None):
    try:
        if _weight_eligible(weight) and _get_backend() is not None:
            return _route(inp, weight, bias)
    except Exception as e:   # never break a render
        print(f"{TAG} fp8-native path failed ({e!r}); delegating to decode path.")
    return _orig(inp, weight, bias)


def install():
    global _orig, _installed
    if _installed:
        return
    # Opt-in only: when disabled, do NOT wrap F.linear at all, so opt-out users pay
    # zero per-call overhead and their F.linear is byte-for-byte untouched by #15.
    # (ASFP8_FP8_EXT is a launch-time flag; enabling it requires a restart.)
    if os.environ.get("ASFP8_FP8_EXT") != "1":
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    import torch.nn.functional as F
    _orig = F.linear
    F.linear = _linear
    _installed = True
    print(f"{TAG} installed (opt-in active; builds on first eligible Linear).")
