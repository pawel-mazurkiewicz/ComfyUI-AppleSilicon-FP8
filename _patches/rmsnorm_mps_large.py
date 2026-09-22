"""Fix: F.rms_norm returns garbage on MPS at large row counts (PiD black image).

The fused MPS kernel silently returns zeros or exploded values once the row count
crosses ~2**22, which NaNs out in bf16. Above the threshold, use the exact manual fp32
formula instead; the fused fast path is kept for normal sizes and every other device.
"""

import torch
import torch.nn.functional as F

TAG = "[AppleSilicon-FP8/rmsnorm]"

# fused is correct at 1.05M rows and garbage at 4.19M, so intervene in between
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
