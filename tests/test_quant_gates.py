"""Kernel seams must route layers comfy forces to full precision only because MPS lacks
the format (comfy marks every int8/fp8/convrot layer that way on MPS), and must still
honour a checkpoint's own full_precision_matrix_mult."""
import json
import os
import sys

import pytest
import torch

_CANDIDATES = [
    os.environ.get("ASFP8_COMFY_PATH"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    os.path.expanduser("~/ComfyUI-Installs/ComfyUI/ComfyUI"),
]
for _c in _CANDIDATES:
    if _c and os.path.isdir(os.path.join(_c, "comfy")) and _c not in sys.path:
        sys.path.insert(0, _c)

ops = pytest.importorskip("comfy.ops")
pytest.importorskip("comfy_kitchen.tensor")

from _patches import _caps, _common  # noqa: E402
from _patches import fp8_linear_kernel_mps as fp8k  # noqa: E402
from _patches import int4_linear_mps as int4  # noqa: E402
from _patches import int8_linear_kernel_mps as int8k  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")

MPS = torch.device("mps")
SENTINEL = object()


def _layer(fmt, weight, scale, n, k, **conf):
    disabled = ops.get_disabled_quant_formats(MPS)
    cls = ops.mixed_precision_ops({}, torch.bfloat16, disabled=disabled)
    lin = cls.Linear(k, n, bias=False, device=MPS, dtype=torch.bfloat16)
    quant = json.dumps({"format": fmt, **conf}).encode()
    sd = {"weight": weight, "weight_scale": scale,
          "comfy_quant": torch.tensor(list(quant), dtype=torch.uint8)}
    lin.load_state_dict(sd, strict=False)
    return lin


def _int8_layer(**conf):
    n, k = 512, 256
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    return _layer("int8_tensorwise", w, torch.tensor(0.01), n, k,
                  convrot=True, convrot_groupsize=256, **conf)


def test_mps_disables_the_formats_our_kernels_serve():
    disabled = ops.get_disabled_quant_formats(MPS)
    assert {"int8_tensorwise", "float8_e4m3fn"} <= set(disabled)


def test_int8_layer_forced_full_precision_still_routes(monkeypatch):
    lin = _int8_layer()
    assert lin._full_precision_mm is True
    assert not _common.full_precision_requested(lin)
    monkeypatch.setattr(_caps, "kernel_ready", lambda *a: True)
    monkeypatch.setattr(int8k, "_int8_linear_kernel", lambda *a: SENTINEL)
    x = torch.randn(4, 256, device=MPS, dtype=torch.bfloat16)
    assert int8k._try_int8_kernel_forward(lin, x) is SENTINEL


def test_int8_checkpoint_full_precision_is_honoured(monkeypatch):
    lin = _int8_layer(full_precision_matrix_mult=True)
    assert _common.full_precision_requested(lin)
    monkeypatch.setattr(_caps, "kernel_ready", lambda *a: True)
    monkeypatch.setattr(int8k, "_int8_linear_kernel", lambda *a: SENTINEL)
    x = torch.randn(4, 256, device=MPS, dtype=torch.bfloat16)
    assert int8k._try_int8_kernel_forward(lin, x) is None


def test_int4_layer_forced_full_precision_still_routes(monkeypatch):
    n, k = 256, 256
    lin = _layer("convrot_w4a4", torch.randint(-128, 128, (n, k // 2), dtype=torch.int8),
                 torch.rand(n, k // 64) * 0.01, n, k, convrot_groupsize=256)
    if getattr(lin.weight, "_layout_cls", None) != "TensorCoreConvRotW4A4Layout":
        pytest.skip("comfy did not build a ConvRot W4A4 weight here")
    assert lin._full_precision_mm is True
    ck = sys.modules["comfy_kitchen.tensor.convrot_w4a4"]
    monkeypatch.setattr(ck, "convrot_w4a4_linear", lambda *a, **k: SENTINEL)
    x = torch.randn(4, k, device=MPS, dtype=torch.bfloat16)
    assert int4._try_int4_forward(lin, x) is SENTINEL


def test_fp8_layer_forced_full_precision_still_routes(monkeypatch):
    n, k = 256, 256
    w = torch.randn(n, k).clamp(-4, 4).to(torch.float8_e4m3fn)
    try:
        lin = _layer("float8_e4m3fn", w, torch.tensor(0.05), n, k)
    except Exception as e:  # MPS fp8 storage support varies by torch build
        pytest.skip(f"fp8 layer did not load on MPS: {e!r}")
    assert lin._full_precision_mm is True
    monkeypatch.setenv("ASFP8_FP8_NATIVE_MIN_DIM", "1")
    monkeypatch.setattr(_caps, "kernel_ready", lambda *a: True)
    monkeypatch.setattr(fp8k, "_fp8_linear_kernel", lambda *a: SENTINEL)
    x = torch.randn(4, k, device=MPS, dtype=torch.bfloat16)
    assert fp8k._try_fp8_kernel_forward(lin, x) is SENTINEL


def test_int8_kernel_skips_large_token_counts(monkeypatch):
    lin = _int8_layer()
    monkeypatch.setattr(_caps, "kernel_ready", lambda *a: True)
    monkeypatch.setattr(int8k, "_int8_linear_kernel", lambda *a: SENTINEL)
    monkeypatch.setenv("ASFP8_INT8_KERNEL_MAX_TOKENS", "64")
    small = torch.randn(64, 256, device=MPS, dtype=torch.bfloat16)
    big = torch.randn(2, 33, 256, device=MPS, dtype=torch.bfloat16)
    assert int8k._try_int8_kernel_forward(lin, small) is SENTINEL
    assert int8k._try_int8_kernel_forward(lin, big) is None
