import os

import pytest
import torch

from conftest import requires_mps

_run_int = os.environ.get("ASFP8_RUN_FP8_EXT_INTEGRATION") == "1"
requires_fp8_ext = pytest.mark.skipif(
    not _run_int, reason="set ASFP8_RUN_FP8_EXT_INTEGRATION=1 (builds the ObjC++ ext, uses GPU)")


@requires_mps
@requires_fp8_ext
@pytest.mark.parametrize("M,K,N", [(256, 3072, 12288), (1024, 12288, 3072), (200, 130, 100)])
def test_fp8_matmul2d_nt_parity(M, K, N):
    os.environ["ASFP8_FP8_EXT"] = "1"
    from _patches.fp8_ext.loader import module
    from _patches._common import decode_fp8
    mod = module()
    assert mod is not None, "extension failed to build"
    torch.manual_seed(1)
    x = (torch.randn(M, K) * 0.3).to(torch.half).to("mps").contiguous()
    torch.manual_seed(0)
    w_fp8 = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn).contiguous()   # [out, in]
    w_u8 = w_fp8.view(torch.uint8).to("mps").contiguous()
    w_mps = w_fp8.to("mps")
    ref = x.float() @ decode_fp8(w_mps, torch.float32).t()                   # x @ Wᵀ
    out = mod.fp8_matmul2d_nt(x, w_u8, N)
    rel = ((out - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
    assert rel < 5e-2, f"NT rel error {rel:.4f} at {(M,K,N)}"
