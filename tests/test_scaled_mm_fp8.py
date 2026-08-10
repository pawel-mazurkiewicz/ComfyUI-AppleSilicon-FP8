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


# --- the F.scaled_mm / aten::_scaled_mm_v2 seam (issue #19) ---


_HAS_V2 = hasattr(torch.nn.functional, "scaled_mm")
requires_v2 = pytest.mark.skipif(not _HAS_V2, reason="requires torch with F.scaled_mm")


@pytest.fixture
def v2_installed(monkeypatch):
    """Install the v2 wrapper over a recorded original, then restore it."""
    calls = []

    def _record(*a, **kw):
        calls.append((a, kw))
        raise NotImplementedError("original F.scaled_mm (no MPS kernel)")

    monkeypatch.setattr(scaled_mm_fp8, "_original_v2", _record)
    if _HAS_V2:
        monkeypatch.setattr(
            torch.nn.functional, "scaled_mm", scaled_mm_fp8._mps_scaled_mm_v2
        )
    return calls


@requires_v2
def test_install_wraps_the_v2_seam(monkeypatch):
    """install() must wrap F.scaled_mm too — wrapping only torch._scaled_mm leaves
    the patch attached to a function comfy_kitchen no longer calls."""
    monkeypatch.setattr(scaled_mm_fp8, "_installed", False)
    monkeypatch.setattr(torch, "_scaled_mm", torch._scaled_mm)
    monkeypatch.setattr(torch.nn.functional, "scaled_mm", torch.nn.functional.scaled_mm)
    scaled_mm_fp8.install()
    assert torch.nn.functional.scaled_mm is scaled_mm_fp8._mps_scaled_mm_v2
    assert scaled_mm_fp8._original_v2 is not None


@requires_mps
@requires_v2
def test_v2_fp8_tensorwise_matches_the_legacy_seam(v2_installed):
    """A plain TensorWise fp8 call through F.scaled_mm must produce exactly what
    the torch._scaled_mm seam produces — same machinery, same numerics."""
    ST = torch.nn.functional.ScalingType
    torch.manual_seed(0)
    a = (torch.randn(64, 128) * 0.3).to(torch.float8_e4m3fn).to("mps")
    w = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).to("mps").t()
    sa = torch.full((1,), 0.7, device="mps")
    sb = torch.full((1,), 1.3, device="mps")
    bias = torch.randn(32, device="mps", dtype=torch.bfloat16)

    got = torch.nn.functional.scaled_mm(
        a, w, scale_a=sa, scale_recipe_a=ST.TensorWise,
        scale_b=sb, scale_recipe_b=ST.TensorWise,
        bias=bias, output_dtype=torch.bfloat16,
    )
    want = scaled_mm_fp8._mps_scaled_mm(
        a, w, out_dtype=torch.bfloat16, scale_a=sa, scale_b=sb, bias=bias,
    )
    assert not v2_installed, "TensorWise fp8 must not reach the original"
    assert torch.equal(got.cpu(), want.cpu())


@requires_mps
@requires_v2
@pytest.mark.parametrize(
    "kwargs_name",
    ["blockwise_swizzled", "list_recipes", "contraction_dim"],
)
def test_v2_passes_microscaling_through_untouched(v2_installed, kwargs_name):
    """MXFP8/NVFP4 recipes and contraction_dim must reach the original unchanged —
    this patch deliberately does not claim them."""
    ST = torch.nn.functional.ScalingType
    SW = torch.nn.functional.SwizzleType
    a = (torch.randn(64, 64) * 0.1).to(torch.float8_e4m3fn).to("mps")
    w = (torch.randn(64, 64) * 0.1).to(torch.float8_e4m3fn).to("mps").t()
    s = torch.ones(1, device="mps")
    variants = {
        "blockwise_swizzled": dict(
            scale_recipe_a=ST.BlockWise1x32, scale_recipe_b=ST.BlockWise1x32,
            swizzle_a=SW.SWIZZLE_32_4_4, swizzle_b=SW.SWIZZLE_32_4_4,
        ),
        "list_recipes": dict(
            scale_recipe_a=[ST.BlockWise1x16, ST.TensorWise],
            scale_recipe_b=[ST.BlockWise1x16, ST.TensorWise],
        ),
        "contraction_dim": dict(
            scale_recipe_a=ST.TensorWise, scale_recipe_b=ST.TensorWise,
            contraction_dim=(1,),
        ),
    }
    with pytest.raises(NotImplementedError):
        torch.nn.functional.scaled_mm(
            a, w, scale_a=s, scale_b=s, output_dtype=torch.bfloat16,
            **variants[kwargs_name],
        )
    assert len(v2_installed) == 1, "call did not reach the original"


@requires_mps
@requires_v2
def test_v2_non_fp8_passes_through(v2_installed):
    ST = torch.nn.functional.ScalingType
    x = torch.randn(32, 32, device="mps", dtype=torch.bfloat16)
    s = torch.ones(1, device="mps")
    with pytest.raises(NotImplementedError):
        torch.nn.functional.scaled_mm(
            x, x, scale_a=s, scale_recipe_a=ST.TensorWise,
            scale_b=s, scale_recipe_b=ST.TensorWise,
        )
    assert len(v2_installed) == 1


@requires_mps
@requires_v2
def test_comfy_kitchen_plain_fp8_entry_point_routes(v2_installed):
    """The live seam: comfy_kitchen's tensor/fp8.py _fp8_scaled_mm is what every
    plain-fp8 Linear actually calls. It must land on our wrapper, not raise."""
    ck_fp8 = pytest.importorskip("comfy_kitchen.tensor.fp8")
    a = (torch.randn(64, 128) * 0.3).to(torch.float8_e4m3fn).to("mps")
    w = (torch.randn(32, 128) * 0.3).to(torch.float8_e4m3fn).to("mps").t()
    s = torch.ones(1, device="mps")
    out = ck_fp8._fp8_scaled_mm(a, w, s, s, out_dtype=torch.bfloat16)
    assert out.device.type == "mps" and out.shape == (64, 32)
    assert not v2_installed, "plain fp8 must not reach the original"
