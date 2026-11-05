"""Native per-row INT8 activation quant must be bit-identical to comfy_kitchen's."""
import pytest
import torch

from conftest import requires_mps

from _patches import _caps
from _patches import int8_quant_mps as mod

_DTYPES = [torch.bfloat16, torch.float16]


@pytest.fixture(autouse=True)
def _fresh_gate(monkeypatch):
    monkeypatch.delenv("ASFP8_INT8_NATIVE_QUANT", raising=False)
    _caps._kernel_ready.pop(mod.NAME, None)
    yield
    _caps._kernel_ready.pop(mod.NAME, None)


def _kq():
    from comfy_kitchen.backends.eager import quantization
    return quantization


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _assert_native_matches(x, monkeypatch):
    kq = _kq()
    assert mod.verify()
    qr, sr = kq.quantize_int8_rowwise(x)

    def no_fallback(*a, **k):
        raise AssertionError("native quant fell back to comfy_kitchen")

    monkeypatch.setattr(kq, "quantize_int8_rowwise", no_fallback)
    q, s = mod.quantize_rowwise(x)
    assert q.dtype == torch.int8 and q.shape == x.shape
    assert s.dtype == torch.float32 and s.shape == (x.shape[0], 1)
    bad = ((q != qr).any(dim=1) | (_bits(s) != _bits(sr)).view(-1)).nonzero().flatten()
    if len(bad):
        r = int(bad[0])
        k = int((q[r] != qr[r]).nonzero().flatten()[0]) if (q[r] != qr[r]).any() else 0
        raise AssertionError(
            f"row {r}: scale {s[r].item()!r} (0x{_bits(s[r]).item() & 0xffffffff:08x}) vs "
            f"{sr[r].item()!r} (0x{_bits(sr[r]).item() & 0xffffffff:08x}); col {k}: "
            f"x={x[r, k].item()!r} (0x{_bits(x[r, k]).item() & 0xffff:04x}) "
            f"q={q[r, k].item()} vs {qr[r, k].item()}"
        )
    assert torch.equal(q, qr) and torch.equal(s, sr)


@requires_mps
@pytest.mark.parametrize("shape", [(1, 256), (7, 2576), (3, 1001), (300, 3072), (4096, 3072),
                                   (40000, 64)])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_quantize_rowwise_matches_kitchen(monkeypatch, dtype, shape):
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(shape, generator=g) * 3).to(dtype).to("mps")
    _assert_native_matches(x, monkeypatch)


@requires_mps
@pytest.mark.parametrize("dtype", _DTYPES)
def test_quantize_rowwise_edge_rows(monkeypatch, dtype):
    K = 1024
    k = torch.arange(K, dtype=torch.float32)
    fmax = torch.finfo(dtype).max
    rows = torch.zeros(7, K)
    rows[1] = (k + 0.5) * 2.0 ** -7
    rows[2] = fmax
    rows[3] = 1e-39 if dtype == torch.bfloat16 else 1e-7
    rows[4, 37] = -2.75
    rows[5] = torch.where(k % 2 == 0, 5.5, -5.5)
    # amax = 127 * 2^-6 makes the scale exactly 2^-6, so every x / s is an exact .5 tie
    rows[6] = torch.where(k % 2 == 0, 1.0, -1.0) * ((k % 127) + 0.5) * 2.0 ** -6
    rows[6, 0] = 127 * 2.0 ** -6
    x = rows.to(dtype).to("mps")
    if dtype == torch.bfloat16:
        assert x[3].abs().max().item() < torch.finfo(dtype).tiny
    _assert_native_matches(x, monkeypatch)


@requires_mps
@pytest.mark.parametrize("dtype,mags", [
    (torch.float16, (1e-2, 1e-3, 1e-4, 1e-5, 3e-6, 1e-6)),   # subnormal fp16 scales
    (torch.bfloat16, (1e-20, 1e-28, 1e-30, 1e-33, 1e-36, 1e-38)),
])
def test_quantize_rowwise_tiny_scales(monkeypatch, dtype, mags):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64 * len(mags), 777, generator=g)
    x *= torch.tensor(mags).repeat_interleave(64).unsqueeze(1)
    _assert_native_matches(x.to(dtype).to("mps"), monkeypatch)


@requires_mps
@pytest.mark.parametrize("dtype", _DTYPES)
def test_quantize_rowwise_every_amax(monkeypatch, dtype):
    # one row per finite non-negative T value as amax; catches a fast-math scale divide
    v = torch.arange(0, 2 ** 15, dtype=torch.int32).to(torch.int16).view(dtype).float()
    v = v[torch.isfinite(v)].unsqueeze(1)
    g = torch.Generator().manual_seed(3)
    x = torch.cat([v, (torch.rand(len(v), 255, generator=g) * 2 - 1) * v], 1)
    _assert_native_matches(x.to(dtype).to("mps"), monkeypatch)


def _spy_kitchen(monkeypatch):
    kq = _kq()
    real, calls = kq.quantize_int8_rowwise, []

    def spy(x, *a, **k):
        calls.append(x)
        return real(x, *a, **k)

    monkeypatch.setattr(kq, "quantize_int8_rowwise", spy)
    return real, calls


def _check_fallback(x, monkeypatch):
    real, calls = _spy_kitchen(monkeypatch)
    q, s = mod.quantize_rowwise(x)
    assert len(calls) == 1
    qr, sr = real(x)
    assert torch.equal(q, qr) and torch.equal(s, sr)


def test_quantize_rowwise_falls_back_on_cpu(monkeypatch):
    _check_fallback(torch.randn(4, 64).to(torch.bfloat16), monkeypatch)


@requires_mps
def test_quantize_rowwise_falls_back_on_fp32(monkeypatch):
    _check_fallback(torch.randn(4, 64, device="mps"), monkeypatch)


@requires_mps
def test_quantize_rowwise_falls_back_when_env_off(monkeypatch):
    monkeypatch.setenv("ASFP8_INT8_NATIVE_QUANT", "off")
    _check_fallback(torch.randn(4, 64, device="mps", dtype=torch.bfloat16), monkeypatch)


@requires_mps
def test_kernel_failure_falls_back_and_latches(monkeypatch, capsys):
    assert mod.verify()
    tries = []

    def boom(x2d):
        tries.append(1)
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(mod, "_native", boom)
    x = torch.randn(4, 64, device="mps", dtype=torch.bfloat16)
    _check_fallback(x, monkeypatch)
    _check_fallback(x, monkeypatch)
    assert len(tries) == 1
    assert mod.TAG in capsys.readouterr().out


@requires_mps
def test_verify_is_cached(monkeypatch):
    assert mod.verify()
    monkeypatch.setattr(mod, "_verify", lambda: pytest.fail("re-verified"))
    assert mod.verify()
