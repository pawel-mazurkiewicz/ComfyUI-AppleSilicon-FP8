import pytest
import torch

from _patches._common import decode_fp8
from conftest import requires_mps

from _patches import scaled_mm_fp8


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


def _cpu_reference(a_fp8, b_fp8, scale_a, scale_b, bias, out_dtype):
    a = a_fp8.to(torch.float32)
    b = b_fp8.to(torch.float32)
    out = a @ b
    if scale_a is not None:
        out = out * scale_a.to(torch.float32)
    if scale_b is not None:
        out = out * scale_b.to(torch.float32)
    if bias is not None:
        out = out + bias.to(torch.float32)
    if out_dtype is not None:
        out = out.to(out_dtype)
    return out


def test_compute_dtype_bf16_for_bf16_out():
    assert scaled_mm_fp8._compute_dtype(torch.bfloat16) == torch.bfloat16
    assert scaled_mm_fp8._compute_dtype(torch.float16) == torch.bfloat16


def test_compute_dtype_f32_for_f32_or_none_out():
    assert scaled_mm_fp8._compute_dtype(torch.float32) == torch.float32
    assert scaled_mm_fp8._compute_dtype(None) == torch.float32


@requires_mps
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_mps_scaled_mm_matches_cpu_reference(out_dtype):
    M, K, N = 64, 128, 32
    torch.manual_seed(0)
    a = (torch.randn(M, K) * 0.3).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn)  # (N,K) so .t() is col-major (K,N)
    scale_a = torch.rand(M, 1) + 0.5
    scale_b = torch.rand(1, N) + 0.5
    bias = torch.randn(N)

    ref = _cpu_reference(a, b.t(), scale_a, scale_b, bias, out_dtype)

    out = scaled_mm_fp8._mps_scaled_mm(
        a.to("mps"), b.t().to("mps"),
        out_dtype=out_dtype,
        scale_a=scale_a.to("mps"), scale_b=scale_b.to("mps"),
        bias=bias.to("mps"),
    ).cpu()

    assert out.dtype == out_dtype
    tol = 0.05 if out_dtype == torch.bfloat16 else 1e-3
    rel = (out.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-9)
    assert rel < tol, f"rel error {rel:.4f} exceeds {tol}"
