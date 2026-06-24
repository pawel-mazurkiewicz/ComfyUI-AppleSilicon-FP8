import pytest
import torch

from conftest import requires_mps

from _patches import sg_gemm


@requires_mps
@pytest.mark.parametrize("m,k,n", [
    (64, 64, 64),       # one threadgroup tile, exact
    (128, 256, 192),    # multi-tile, exact multiples
    (200, 130, 100),    # ragged edges (not multiples of tile)
])
def test_sg_matmul_matches_reference(m, k, n):
    if not sg_gemm.available():
        pytest.skip("sg_gemm kernel did not compile on this SDK")
    torch.manual_seed(0)
    a = (torch.randn(m, k) * 0.3).to(torch.bfloat16).to("mps")
    b = (torch.randn(k, n) * 0.3).to(torch.bfloat16).to("mps")
    ref = a.float() @ b.float()
    out = sg_gemm.sg_matmul(a, b).cpu()
    rel = (out.float() - ref.cpu()).abs().max() / (ref.cpu().abs().max() + 1e-9)
    assert rel < 5e-2, f"rel error {rel:.4f} for shape {(m,k,n)}"


@requires_mps
def test_self_check_ok():
    if not sg_gemm.available():
        pytest.skip("sg_gemm kernel did not compile on this SDK")
    assert sg_gemm.self_check_ok() is True
