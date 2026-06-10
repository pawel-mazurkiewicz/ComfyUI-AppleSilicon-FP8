"""Fix: torch.nn.functional.rms_norm returns garbage on MPS at large row counts.

The fused MPS rms_norm kernel silently produces wrong output (observed: zeros or
wildly exploded values) once the number of normalization rows
(input.numel() // prod(normalized_shape)) crosses ~2**22 (~4.19M). Below that it
is correct; the manual formula is correct at every size. (Same family as the
fused-SDPA-large-sequence MPS bug.)

This breaks PiD (Pixel Diffusion Decoder): its pixel blocks RMSNorm a
[BL, P2, pixel_dim] tensor whose row count is (out_px/16)**2 * 256 —
  1024px -> 1.05M rows (fine), 2048px -> 4.19M (broken), 4096px -> 16.8M (broken).
Garbage RMSNorm -> activation explosion -> NaN in bf16 -> fully black image.

Fix: on MPS, when the row count is large, compute rms_norm with the exact manual
formula in fp32 (x * rsqrt(mean(x^2) + eps) * weight). The fused fast path is kept
for normal sizes and for every non-MPS device, so there's no perf cost elsewhere.
"""

import torch
import torch.nn.functional as F

TAG = "[AppleSilicon-FP8/rmsnorm]"

# Fused confirmed correct at 1.05M rows, garbage at 4.19M. Intervene above 2.1M:
# catches the broken regime, leaves all normal usage on the fast fused path.
_THRESHOLD = 1 << 21

_orig = None
_installed = False


def _rms_norm(input, normalized_shape, weight=None, eps=None):
    rows = 1
    for d in normalized_shape:
        rows *= d
    rows = input.numel() // max(rows, 1)

    if input.device.type != "mps" or rows <= _THRESHOLD:
        return _orig(input, normalized_shape, weight, eps)

    # Manual rms_norm in fp32 — exact at any size.
    ndims = len(normalized_shape)
    dims = tuple(range(input.dim() - ndims, input.dim()))
    e = eps if eps is not None else torch.finfo(input.dtype).eps
    xf = input.float()
    var = xf.pow(2).mean(dims, keepdim=True)
    out = (xf * torch.rsqrt(var + e)).to(input.dtype)
    if weight is not None:
        out = out * weight
    return out


def install():
    global _orig, _installed
    if _installed:
        return
    _orig = F.rms_norm
    F.rms_norm = _rms_norm
    torch.nn.functional.rms_norm = _rms_norm
    _installed = True
    print(f"{TAG} F.rms_norm uses manual fp32 path on MPS for >2^21 rows (PiD black-image fix).")
