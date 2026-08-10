import pytest
import torch

from conftest import requires_mps

from _patches import tensor_to_fp8


@pytest.fixture(autouse=True)
def _installed(monkeypatch):
    """Install the patch onto a pristine torch.Tensor, restored after each test."""
    monkeypatch.setattr(tensor_to_fp8, "_installed", False)
    monkeypatch.setattr(torch.Tensor, "to", torch.Tensor.to)
    for name in ("float", "half", "bfloat16"):
        monkeypatch.setattr(torch.Tensor, name, getattr(torch.Tensor, name))
    tensor_to_fp8.install()


@requires_mps
@pytest.mark.parametrize(
    "method,want_dtype",
    [
        ("float", torch.float32),
        ("half", torch.float16),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_fp8_dtype_shortcuts_work_on_mps(method, want_dtype):
    """`.float()` and friends bypass Tensor.to entirely, so patching only .to()
    left them raising on MPS (issue #16: `s_rel.float()` on an fp8 group-scale
    tensor -> RuntimeError: Undefined type Float8_e4m3fn)."""
    src = torch.randn(32) * 0.5
    t = src.to(torch.float8_e4m3fn).to("mps")

    got = getattr(t, method)()

    assert got.dtype == want_dtype
    assert got.device.type == "mps"
    want = src.to(torch.float8_e4m3fn).to(torch.float32)
    assert torch.equal(got.cpu().to(torch.float32), want)


@requires_mps
def test_fp8_shortcut_is_exact_over_every_byte():
    raw = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to("mps")
    got = raw.float().cpu()
    want = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    finite = torch.isfinite(want)
    assert torch.equal(got[finite], want[finite])


@requires_mps
def test_non_fp8_shortcuts_are_untouched():
    t = torch.randn(8, device="mps", dtype=torch.float32)
    assert t.half().dtype == torch.float16
    assert t.bfloat16().dtype == torch.bfloat16
    assert torch.equal(t.float(), t)


@requires_mps
def test_double_still_raises_on_mps():
    """MPS has no float64, so .double() has nothing to rescue — it must keep
    raising torch's own error rather than being silently rerouted.

    The exception type moves between torch versions (fp8 casts raise TypeError on
    2.11 and RuntimeError on 2.14-dev), so match on the message instead to pin
    that this is still the float64-unsupported path.
    """
    t = (torch.randn(8) * 0.5).to(torch.float8_e4m3fn).to("mps")
    with pytest.raises((TypeError, RuntimeError), match="float64"):
        t.double()


def test_cpu_fp8_shortcut_delegates_to_original():
    """CPU can cast fp8 natively; the patch must not intercept it."""
    t = (torch.randn(8) * 0.5).to(torch.float8_e4m3fn)
    assert torch.equal(t.float(), t.to(torch.float32))


@requires_mps
def test_w4a8_grouped_dequant_with_fp8_group_scales():
    """The issue #16 shape: comfy_kitchen's _dequant_int4_grouped_to_int8 does
    `s_rel.float()` on group scales that the W4A8-mixed checkpoint stores as fp8."""
    n, k, group_size = 8, 64, 16
    groups = k // group_size
    torch.manual_seed(0)

    qdata = torch.randint(-128, 128, (n, k // 2), dtype=torch.int8, device="mps")
    s_rel = (torch.rand(n, groups) * 0.5 + 0.1).to(torch.float8_e4m3fn).to("mps")

    def dequant(qdata, s_rel, group_size):
        n, k_half = qdata.shape
        k = k_half * 2
        groups = k // group_size
        packed = qdata.to(torch.int32) & 0xFF
        quantized = torch.empty(n, k, dtype=torch.int32, device=qdata.device)
        quantized[:, 0::2] = packed & 0xF
        quantized[:, 1::2] = (packed >> 4) & 0xF
        values = quantized.float() - 8.0
        values = values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
        return values.view(n, k).round().clamp_(-127, 127).to(torch.int8)

    got = dequant(qdata, s_rel, group_size)

    assert got.dtype == torch.int8 and got.shape == (n, k)
    want = dequant(qdata.cpu(), s_rel.cpu(), group_size)
    assert torch.equal(got.cpu(), want)
