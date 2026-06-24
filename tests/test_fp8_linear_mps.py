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


# --- no-GPU unit tests: eligibility predicate + wrapper routing ---

from _patches import fp8_linear_mps as fl


class _FakeW:
    def __init__(self, shape, dtype, dev="mps"):
        self.shape = shape
        self.dtype = dtype
        class _D:
            type = dev
        self.device = _D()
    def dim(self):
        return len(self.shape)


def test_eligible_large_mlp_fp8(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    w = _FakeW((12288, 3072), torch.float8_e4m3fn)
    assert fl._weight_eligible(w) is True


def test_ineligible_square_below_threshold(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    w = _FakeW((3072, 3072), torch.float8_e4m3fn)
    assert fl._weight_eligible(w) is False


def test_ineligible_non_fp8(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    w = _FakeW((12288, 3072), torch.bfloat16)
    assert fl._weight_eligible(w) is False


def test_ineligible_when_opt_out(monkeypatch):
    monkeypatch.delenv("ASFP8_FP8_EXT", raising=False)
    w = _FakeW((12288, 3072), torch.float8_e4m3fn)
    assert fl._weight_eligible(w) is False


def test_threshold_env_override(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    monkeypatch.setenv("ASFP8_FP8_EXT_MIN_DIM", "4096")
    w = _FakeW((4096, 1024), torch.float8_e4m3fn)
    assert fl._weight_eligible(w) is True


def test_wrapper_delegates_when_ineligible(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(fl, "_orig", lambda inp, w, b=None: sentinel)
    monkeypatch.setattr(fl, "_weight_eligible", lambda w: False)
    assert fl._linear(object(), object(), None) is sentinel


def test_wrapper_routes_when_eligible(monkeypatch):
    monkeypatch.setattr(fl, "_orig", lambda *a, **k: "DELEGATED")
    monkeypatch.setattr(fl, "_weight_eligible", lambda w: True)
    monkeypatch.setattr(fl, "_get_backend", lambda: object())
    monkeypatch.setattr(fl, "_route", lambda inp, w, bias: "ROUTED")
    assert fl._linear(object(), object(), None) == "ROUTED"


def test_wrapper_delegates_on_route_exception(monkeypatch):
    monkeypatch.setattr(fl, "_orig", lambda *a, **k: "DELEGATED")
    monkeypatch.setattr(fl, "_weight_eligible", lambda w: True)
    monkeypatch.setattr(fl, "_get_backend", lambda: object())
    def boom(inp, w, bias):
        raise RuntimeError("kernel blew up")
    monkeypatch.setattr(fl, "_route", boom)
    assert fl._linear(object(), object(), None) == "DELEGATED"
