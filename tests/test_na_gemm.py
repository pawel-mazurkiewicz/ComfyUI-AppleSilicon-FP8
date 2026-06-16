import pytest
import torch
from conftest import requires_mps

from _patches import na_gemm


@requires_mps
def test_na_available_is_bool():
    assert isinstance(na_gemm.available(), bool)


@requires_mps
def test_na_matmul_matches_reference_when_available():
    if not na_gemm.available():
        pytest.skip("NA matmul2d not available on this OS/SDK/PyTorch build")
    M, K, N = 64, 256, 96
    torch.manual_seed(0)
    a = (torch.randn(M, K) * 0.5).to(torch.bfloat16)
    b = (torch.randn(K, N) * 0.5).to(torch.bfloat16)
    ref = a.float() @ b.float()
    out = na_gemm.na_matmul(a.to("mps").contiguous(), b.to("mps").contiguous()).cpu()
    assert out.dtype == torch.float32
    rel = (out - ref).abs().max() / (ref.abs().max() + 1e-9)
    assert rel < 5e-2, f"rel error {rel:.4f}"


@requires_mps
def test_self_check_caches_and_is_bool():
    v1 = na_gemm.self_check_ok()
    v2 = na_gemm.self_check_ok()
    assert isinstance(v1, bool) and v1 == v2
