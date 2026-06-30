import pytest
import torch

from conftest import requires_mps  # repo-provided marker (tests/conftest.py)


@requires_mps
@pytest.mark.parametrize("H,W,Cin,kh,kw,s,p", [(16, 16, 8, 3, 3, 1, 1),
                                               (15, 17, 6, 3, 3, 2, 1),
                                               (8, 8, 4, 1, 1, 1, 0)])
def test_im2col_2d_matches_unfold(H, W, Cin, kh, kw, s, p):
    from _patches.conv_im2col_mps import _im2col_2d_full  # test helper: full (untiled) im2col
    torch.manual_seed(0)
    x = torch.randn(1, Cin, H, W, device="mps", dtype=torch.float16)
    # F.unfold gives [N, Cin*kh*kw, L] with the SAME (c,ki,kj) ordering as our K index.
    ref = torch.nn.functional.unfold(x.float(), (kh, kw), stride=s, padding=p)  # [1, K, P]
    ref = ref[0].t().contiguous()  # [P, K]
    A = _im2col_2d_full(x, kh, kw, s, p).float()  # [P, K]
    torch.mps.synchronize()
    assert A.shape == ref.shape, (A.shape, ref.shape)
    assert (A - ref).abs().max().item() < 1e-2
