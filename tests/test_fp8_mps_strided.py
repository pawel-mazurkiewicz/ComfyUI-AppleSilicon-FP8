import pytest
import torch

from conftest import requires_mps

from _patches import fp8_mps_strided as fs


# --- _wrap logic (no MPS needed): guard, CPU-bounce, and don't-mask-real-errors ---


class _Fake:
    """Stand-in tensor: fp8-on-mps that fails natively, succeeds via .cpu()."""
    def __init__(self, dtype=torch.float8_e4m3fn, dev="mps", on_cpu=False):
        self.dtype = dtype
        self._on_cpu = on_cpu
        class _D:
            type = "cpu" if on_cpu else dev
        self.device = _D()
    def cpu(self):
        return _Fake(self.dtype, on_cpu=True)
    def to(self, *a, **k):
        return self


def test_passthrough_non_fp8():
    calls = []
    def orig(self, *a, **k):
        calls.append(self)
        return "OUT"
    f = _Fake(dtype=torch.float32)
    assert fs._wrap(orig)(f) == "OUT"
    assert calls == [f]  # called once, directly, no bounce


def test_cpu_bounce_on_float8_error():
    seen = []
    def orig(self, *a, **k):
        seen.append(self._on_cpu)
        if not self._on_cpu:
            raise RuntimeError("Undefined type Float8_e4m3fn")
        return _Fake(on_cpu=True)  # the .to("mps") below runs on this
    fs._wrap(orig)(_Fake())
    assert seen == [False, True]  # tried MPS, then CPU


def test_real_runtime_error_not_masked():
    def orig(self, *a, **k):
        raise RuntimeError("some unrelated MPS failure")
    with pytest.raises(RuntimeError, match="unrelated"):
        fs._wrap(orig)(_Fake())


# --- real behaviour on MPS: the exact crash sites ---


@requires_mps
def test_reshape_after_transpose_fp8_mps():
    fs.install()
    x = torch.arange(512, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(4, 128).to("mps")
    blocks = x.reshape(-1, 4, 32, 4).transpose(1, 2)          # non-contiguous fp8 on mps
    out = blocks.reshape(-1, 32, 16)                          # used to raise "Undefined type Float8_e4m3fn"
    assert out.device.type == "mps" and out.dtype == torch.float8_e4m3fn
    ref = blocks.cpu().reshape(-1, 32, 16)
    assert torch.equal(out.cpu().view(torch.uint8), ref.view(torch.uint8))  # bit-exact


@requires_mps
def test_contiguous_on_strided_fp8_mps():
    fs.install()
    x = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(16, 16).to("mps")
    strided = x.t()                                           # non-contiguous fp8
    out = strided.contiguous()
    assert out.device.type == "mps" and out.is_contiguous()
    assert torch.equal(out.cpu().view(torch.uint8), strided.cpu().contiguous().view(torch.uint8))


@requires_mps
def test_non_fp8_reshape_unaffected():
    fs.install()
    y = torch.randn(4, 4, device="mps")
    assert tuple(y.t().reshape(16).shape) == (16,)           # ordinary path still works
