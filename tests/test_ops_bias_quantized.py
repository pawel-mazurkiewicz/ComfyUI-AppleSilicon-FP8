"""Patch #6 (ops_bias) must not intercept comfy_kitchen QuantizedTensor weights.

Regression tests for issue #9: loading the MiniMax H3 int8 text encoder raised

    NoCapableBackendError: No backend can handle 'dequantize_int8_embedding':
    eager: q: dtype torch.bfloat16 not in {torch.int8}

because our cast_bias_weight override dequantized the int8 QuantizedTensor to a
plain bf16 tensor.  comfy's Embedding.forward_comfy_cast_weights needs the
wrapper back so it can reach the raw int8 storage for a per-row gather.

Native cast_bias_weight handles QuantizedTensors of any layout fine on MPS; it
only fails on RAW fp8 tensors, which is the case this patch exists for.
"""
import inspect
import os
import sys

import pytest
import torch

# ComfyUI-desktop keeps the code tree outside the repo venv, so comfy is not
# importable by default (same convention as test_rope_fast_comfy.py).
_CANDIDATES = [
    os.environ.get("ASFP8_COMFY_PATH"),
    "/Users/pawelma/ComfyUI-Installs/ComfyUI/ComfyUI",
]
for _c in _CANDIDATES:
    if _c and os.path.isdir(os.path.join(_c, "comfy")) and _c not in sys.path:
        sys.path.insert(0, _c)

ops = pytest.importorskip("comfy.ops")
pytest.importorskip("comfy.quant_ops")
QuantizedTensor = pytest.importorskip("comfy_kitchen.tensor").QuantizedTensor

from _patches import comfykitchen_fp8 as ck  # noqa: E402
from _patches import ops_bias_fp8 as m  # noqa: E402

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs MPS"
)


class _FakeLayer:
    """Minimal stand-in for a comfy manual-cast layer."""

    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias
        self.weight_function = []
        self.bias_function = []


@pytest.fixture
def patched():
    """Install the ops_bias override, then restore the real one.

    comfykitchen_fp8 installs before ops_bias at runtime and is what makes the
    fp8 dequantize path MPS-safe, so it has to be in place here too.
    """
    ck.install()
    original = ops.cast_bias_weight
    m._installed = False
    m.install()
    assert ops.cast_bias_weight is not original, "patch did not install"
    yield
    ops.cast_bias_weight = original
    m._installed = False


def _quantized(layout, **kwargs):
    src = torch.randn(64, 32, dtype=torch.bfloat16)
    return QuantizedTensor.from_float(src, layout, **kwargs).to(device="mps")


@requires_mps
def test_int8_quantized_weight_keeps_its_wrapper(patched):
    """int8 QuantizedTensor must survive cast_bias_weight as a QuantizedTensor."""
    weight = _quantized("TensorWiseINT8Layout")
    layer = _FakeLayer(weight)

    w, _ = ops.cast_bias_weight(layer, device=torch.device("mps"), dtype=weight.dtype)

    assert isinstance(w, QuantizedTensor), (
        f"wrapper stripped: got {type(w).__name__} dtype={w.dtype}; "
        "comfy's Embedding path needs the raw int8 storage"
    )
    assert w.storage_dtype == torch.int8


@requires_mps
def test_fp8_quantized_weight_is_decoded(patched):
    """fp8 storage must still be rescued: MPS cannot gather on fp8 qdata.

    Guards the int8 fix from over-reaching into a blanket "delegate every
    QuantizedTensor", which would push raw fp8 into F.embedding and raise.
    """
    weight = _quantized("TensorCoreFP8Layout", scale=torch.tensor(1.0))
    layer = _FakeLayer(weight)

    w, _ = ops.cast_bias_weight(layer, device=torch.device("mps"), dtype=weight.dtype)

    assert not isinstance(w, QuantizedTensor)
    assert w.dtype == torch.bfloat16


@requires_mps
def test_real_fp8_embedding_runs_on_mps(patched, monkeypatch):
    """End-to-end: an fp8 embedding must survive its forward on MPS.

    comfy hands the qdata straight to F.embedding, and MPS has no fp8 gather,
    so our patch has to have decoded it to bf16 first.
    """
    MP = ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    layer = MP.Embedding(16, 32, device="cpu", dtype=torch.bfloat16)
    src = torch.randn(16, 32, dtype=torch.bfloat16)
    weight = QuantizedTensor.from_float(
        src, "TensorCoreFP8Layout", scale=torch.tensor(1.0)
    )
    layer.weight = torch.nn.Parameter(weight.to(device="mps"), requires_grad=False)
    layer.quant_format = "float8_e4m3fn"
    layer.layout_type = "TensorCoreFP8Layout"
    layer.to("mps")

    seen = []
    real_embedding = torch.nn.functional.embedding

    def spy(input, weight, *args, **kwargs):
        seen.append(weight.dtype)
        return real_embedding(input, weight, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "embedding", spy)
    out = layer.forward_comfy_cast_weights(torch.tensor([0, 3, 5], device="mps"))

    assert seen == [torch.bfloat16], f"embedding looked up in {seen}, expected bf16"
    assert out.shape == (3, 32)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.cpu().float(), weight.dequantize()[[0, 3, 5]].float())


@requires_mps
def test_int8_embedding_dequantizes_without_backend_error(patched):
    """The exact issue #9 crash: comfy >=0.30 routes int8 embeddings through
    dequantize_int8_embedding, which rejects anything but int8 storage.

    Skipped on older comfy, which has no int8_tensorwise branch.
    """
    MP = ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    layer = MP.Embedding(16, 32, device="cpu", dtype=torch.bfloat16)
    if "int8_tensorwise" not in inspect.getsource(ops.mixed_precision_ops):
        pytest.skip("comfy too old for the int8_tensorwise embedding path")

    src = torch.randn(16, 32, dtype=torch.bfloat16)
    weight = QuantizedTensor.from_float(src, "TensorWiseINT8Layout")
    layer.weight = torch.nn.Parameter(weight.to(device="mps"), requires_grad=False)
    layer.quant_format = "int8_tensorwise"
    layer.layout_type = "TensorWiseINT8Layout"
    layer.to("mps")

    out = layer.forward_comfy_cast_weights(torch.tensor([0, 3, 5], device="mps"))

    assert out.shape == (3, 32)
    torch.testing.assert_close(
        out.cpu().float(), weight.dequantize()[[0, 3, 5]].float(), rtol=0, atol=0
    )


@requires_mps
def test_raw_fp8_weight_is_still_decoded(patched):
    """The case the patch exists for: native raises, we must decode to bf16."""
    src = torch.randn(64, 32, dtype=torch.bfloat16)
    weight = src.to(torch.float8_e4m3fn).to("mps")
    layer = _FakeLayer(weight)

    w, _ = ops.cast_bias_weight(layer, device=torch.device("mps"), dtype=torch.bfloat16)

    assert w.dtype == torch.bfloat16
    assert w.device.type == "mps"
    torch.testing.assert_close(w.cpu().float(), src.to(torch.float8_e4m3fn).float())


@requires_mps
def test_int8_weight_survives_an_fp8_bias(patched):
    """The rescue is decided per layer but must be applied per parameter.

    An fp8 bias must not drag an int8 weight through dequantize() -- that is the
    issue #9 failure reached by a different route.
    """
    weight = _quantized("TensorWiseINT8Layout")
    bias = torch.randn(64, dtype=torch.bfloat16).to(torch.float8_e4m3fn).to("mps")
    layer = _FakeLayer(weight, bias)

    w, b = ops.cast_bias_weight(layer, device=torch.device("mps"), dtype=weight.dtype)

    assert isinstance(w, QuantizedTensor), f"wrapper stripped by the bias: {type(w).__name__}"
    assert b.dtype == torch.bfloat16, "the fp8 bias still needs decoding"
