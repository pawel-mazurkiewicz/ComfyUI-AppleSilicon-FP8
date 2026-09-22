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
    """The dtype shortcuts work on MPS: they bypass Tensor.to entirely (#16)."""
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
    """.double() still raises on MPS, which has no float64 to rescue it into.

    Matched on the message, since the exception type moves between torch versions.
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
    """A W4A8 grouped dequant works with the fp8 group scales the checkpoint stores (#16)."""
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


@requires_mps
@pytest.mark.parametrize("method", ["float", "half", "bfloat16"])
def test_memory_format_keyword_is_honoured(method):
    """memory_format is forwarded, not dropped, by the fp8 path."""
    t = (torch.randn(2, 3, 4, 5) * 0.5).to(torch.float8_e4m3fn).to("mps")
    got = getattr(t, method)(memory_format=torch.channels_last)
    assert got.is_contiguous(memory_format=torch.channels_last)


@requires_mps
@pytest.mark.parametrize("method", ["float", "half", "bfloat16"])
def test_positional_memory_format_rejected_like_stock_torch(method):
    """A positional memory_format is rejected as stock torch rejects it, not passed on
    to .to() as non_blocking."""
    t8 = (torch.randn(2, 3, 4, 5) * 0.5).to(torch.float8_e4m3fn).to("mps")
    t32 = torch.randn(2, 3, 4, 5, device="mps")
    for t in (t8, t32):
        with pytest.raises(TypeError):
            getattr(t, method)(torch.channels_last)
