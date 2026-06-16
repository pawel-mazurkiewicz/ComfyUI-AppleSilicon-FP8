import torch

from _patches._common import decode_fp8


def test_decode_fp8_bf16_is_bit_exact_with_cpu_cast():
    # Every FP8 byte must decode to the same value bf16's exact cast gives.
    raw = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
    want = raw.to(torch.bfloat16)
    got = decode_fp8(raw, torch.bfloat16)
    # NaN bytes compare unequal; mask them and compare the rest exactly.
    finite = torch.isfinite(want)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got[finite], want[finite])


def test_decode_fp8_handles_non_contiguous_input():
    base = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(16, 16)
    transposed = base.t()  # non-contiguous
    got = decode_fp8(transposed, torch.float32)
    want = transposed.contiguous().to(torch.float32)
    finite = torch.isfinite(want)
    assert torch.equal(got[finite], want[finite])
