"""Shared FP8 helpers. MPS has no FP8 casts, so decode via a 256-entry uint8 LUT.

The table is built on CPU (where the cast works) and gathered on-device, which is
bit-exact with a real FP8->float cast.
"""

import torch

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_lut_cache = {}


def fp8_to_float_lut(dtype, device, out_dtype=torch.float32):
    """Return a cached 256-entry LUT (`out_dtype`) for FP8 `dtype`, on `device`."""
    key = (dtype, out_dtype, device.type, getattr(device, "index", None))
    lut = _lut_cache.get(key)
    if lut is None:
        lut = torch.arange(256, dtype=torch.uint8).view(dtype).to(out_dtype).to(device)
        _lut_cache[key] = lut
    return lut


def decode_fp8(t, out_dtype=torch.float32):
    """Decode an FP8 tensor to `out_dtype` on its own device (MPS-safe)."""
    device = t.device
    lut = fp8_to_float_lut(t.dtype, device, out_dtype)
    if t.is_contiguous():
        idx = t.view(torch.uint8).to(torch.long)
    else:
        # MPS can't .contiguous() an FP8 tensor, so do it on CPU
        idx = t.cpu().contiguous().view(torch.uint8).to(torch.long).to(device)
    return lut[idx]
