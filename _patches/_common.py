"""Shared helpers: MPS-safe FP8 decoding via a lookup table.

PyTorch's MPS backend has no FP8 dtype support, so you cannot cast a
float8_e4m3fn / float8_e5m2 tensor to/from float on the GPU. But you *can*:
  - create FP8 tensors on CPU and move them to MPS,
  - bit-view an FP8 tensor as uint8 on MPS,
  - gather/index on MPS.

So we build a 256-entry table mapping every possible FP8 byte to its float
value (decoded on CPU, where the cast works), move that tiny table to MPS once,
then decode any FP8 tensor with a gather: lut[x.view(uint8)]. This is bit-exact
with a real FP8->float cast and runs entirely on the GPU.
"""

import torch

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_lut_cache = {}


def fp8_to_float_lut(dtype, device, out_dtype=torch.float32):
    """Return a cached 256-entry LUT (`out_dtype`) for FP8 `dtype`, on `device`."""
    key = (dtype, out_dtype, device.type, getattr(device, "index", None))
    lut = _lut_cache.get(key)
    if lut is None:
        # Decode every FP8 byte on CPU (where the cast works), then move to device.
        lut = torch.arange(256, dtype=torch.uint8).view(dtype).to(out_dtype).to(device)
        _lut_cache[key] = lut
    return lut


def decode_fp8(t, out_dtype=torch.float32):
    """Decode an FP8 tensor to `out_dtype` on its own device (MPS-safe).

    bf16 and float32 both represent every FP8 value exactly, so either target is
    bit-exact with a real FP8->float cast.
    """
    lut = fp8_to_float_lut(t.dtype, t.device, out_dtype)
    idx = t.contiguous().view(torch.uint8).to(torch.long)
    return lut[idx]
