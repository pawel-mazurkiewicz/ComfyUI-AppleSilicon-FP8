import os

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


# --- opt-in fp8-native fast path: eligibility predicate + routing (no GPU) ---


class _FakeT:
    """Minimal tensor stand-in for the pure-predicate path of _fast_eligible."""
    def __init__(self, shape, dtype=torch.float8_e4m3fn, dev="mps"):
        self.shape = shape
        self.dtype = dtype
        class _D:
            type = dev
        self.device = _D()
    def dim(self):
        return len(self.shape)


def _operands(K=12288, N=3072, M=64, dtype=torch.float8_e4m3fn, dev="mps"):
    # _scaled_mm layout: input [M,K], other [K,N] (W.t() of a [N,K] weight).
    return _FakeT((M, K), dtype, dev), _FakeT((K, N), dtype, dev)


def test_fast_eligible_large_mlp(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    inp, other = _operands(K=12288, N=3072)
    assert scaled_mm_fp8._fast_eligible(inp, other) is True


def test_fast_ineligible_below_threshold(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    inp, other = _operands(K=3072, N=3072)
    assert scaled_mm_fp8._fast_eligible(inp, other) is False


def test_fast_ineligible_non_fp8(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    inp, other = _operands(dtype=torch.bfloat16)
    assert scaled_mm_fp8._fast_eligible(inp, other) is False


def test_fast_ineligible_mixed_fp8_e5m2(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    inp = _FakeT((64, 12288), torch.float8_e4m3fn)
    other = _FakeT((12288, 3072), torch.float8_e5m2)
    assert scaled_mm_fp8._fast_eligible(inp, other) is False


def test_fast_ineligible_non_mps(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    inp, other = _operands(dev="cpu")
    assert scaled_mm_fp8._fast_eligible(inp, other) is False


def test_fast_ineligible_when_explicitly_off(monkeypatch):
    # Explicit opt-out wins over capability + shape, even on supported hardware.
    from _patches import _caps
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: True)
    monkeypatch.setenv("ASFP8_FP8_EXT", "off")
    inp, other = _operands(K=12288, N=3072)
    assert scaled_mm_fp8._fast_eligible(inp, other) is False


def test_default_on_when_capable(monkeypatch):
    # DEFAULT ON: unset + Tier-B capable -> eligible (no explicit flag needed).
    from _patches import _caps
    monkeypatch.delenv("ASFP8_FP8_EXT", raising=False)
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: True)
    inp, other = _operands(K=12288, N=3072)
    assert scaled_mm_fp8._fast_eligible(inp, other) is True


def test_default_off_when_not_capable(monkeypatch):
    # DEFAULT gated: unset + unsupported hardware -> inert (the promise).
    from _patches import _caps
    monkeypatch.delenv("ASFP8_FP8_EXT", raising=False)
    monkeypatch.setattr(_caps, "tier_b_ready", lambda: False)
    inp, other = _operands(K=12288, N=3072)
    assert scaled_mm_fp8._fast_eligible(inp, other) is False


def test_fast_eligible_threshold_env_override(monkeypatch):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    monkeypatch.setenv("ASFP8_FP8_EXT_MIN_DIM", "4096")
    inp, other = _operands(K=4096, N=1024)
    assert scaled_mm_fp8._fast_eligible(inp, other) is True


@requires_mps
def test_fast_path_taken_when_eligible(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(scaled_mm_fp8, "_fast_eligible", lambda i, o: True)
    monkeypatch.setattr(
        scaled_mm_fp8, "_fast_route",
        lambda inp, other, sa, sb, sr, b, od: sentinel,
    )
    a = (torch.randn(8, 16) * 0.3).to(torch.float8_e4m3fn).to("mps")
    b = (torch.randn(4, 16) * 0.3).to(torch.float8_e4m3fn).to("mps")
    out = scaled_mm_fp8._mps_scaled_mm(a, b.t(), out_dtype=torch.float32)
    assert out is sentinel


@requires_mps
def test_fast_path_delegates_to_decode_on_exception(monkeypatch):
    monkeypatch.setattr(scaled_mm_fp8, "_fast_eligible", lambda i, o: True)

    def boom(*a, **k):
        raise RuntimeError("kernel blew up")

    monkeypatch.setattr(scaled_mm_fp8, "_fast_route", boom)

    M, K, N = 32, 64, 16
    torch.manual_seed(0)
    a = (torch.randn(M, K) * 0.3).to(torch.float8_e4m3fn)
    b = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn)
    scale_a = torch.rand(M, 1) + 0.5
    scale_b = torch.rand(1, N) + 0.5
    ref = _cpu_reference(a, b.t(), scale_a, scale_b, None, torch.float32)
    out = scaled_mm_fp8._mps_scaled_mm(
        a.to("mps"), b.t().to("mps"),
        out_dtype=torch.float32,
        scale_a=scale_a.to("mps"), scale_b=scale_b.to("mps"),
    ).cpu()
    rel = (out - ref).abs().max() / (ref.abs().max() + 1e-9)
    assert rel < 1e-3, f"fallback rel {rel:.4f} — did not delegate to decode"


# --- opt-in GPU integration: fp8-native scaled path parity vs decode ---

_run_int = os.environ.get("ASFP8_RUN_FP8_EXT_INTEGRATION") == "1"
requires_fp8_ext = pytest.mark.skipif(
    not _run_int, reason="set ASFP8_RUN_FP8_EXT_INTEGRATION=1 (builds the ObjC++ ext, uses GPU)")


@requires_mps
@requires_fp8_ext
@pytest.mark.parametrize("M,K,N", [(1024, 3072, 12288), (4096, 12288, 3072)])
def test_fast_route_parity_with_decode(monkeypatch, M, K, N):
    monkeypatch.setenv("ASFP8_FP8_EXT", "1")
    monkeypatch.setenv("ASFP8_FP8_EXT_MIN_DIM", "2048")
    # fresh backend per run so the self-check + build are exercised
    scaled_mm_fp8._backend = None
    scaled_mm_fp8._self_checked = False

    torch.manual_seed(1)
    a = (torch.randn(M, K) * 0.3).to(torch.float8_e4m3fn)
    torch.manual_seed(0)
    w = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn)   # [N,K] weight
    scale_a = (torch.rand(M, 1) * 0.5 + 0.5).to("mps")
    scale_b = (torch.rand(1, N) * 0.5 + 0.5).to("mps")
    bias = torch.randn(N).to("mps")

    a_mps, w_mps = a.to("mps"), w.to("mps")
    ref = (decode_fp8(a_mps, torch.float32) @ decode_fp8(w_mps, torch.float32).t())
    ref = ref * scale_a * scale_b + bias.to(torch.float32)

    out = scaled_mm_fp8._mps_scaled_mm(
        a_mps, w_mps.t(),
        out_dtype=torch.float32,
        scale_a=scale_a, scale_b=scale_b, bias=bias,
    )
    assert scaled_mm_fp8._backend not in (None, False), "fast path did not engage"
    rel = ((out - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
    assert rel < 5e-2, f"fp8-native scaled rel {rel:.4f} at {(M,K,N)}"
