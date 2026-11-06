"""Community INT8 checkpoints whose comfy_quant JSON has no "format" must reach comfy's
loader with int8_tensorwise added, and nothing else may be touched.

No MPS and no ComfyUI needed: comfy.ops is stood in by a SimpleNamespace.
"""
import json
import sys
import types

import pytest
import torch

from _patches import quant_format_default as m

CONVROT = {"convrot": True, "convrot_groupsize": 256, "per_row": True}


def enc(d):
    return torch.frombuffer(bytearray(json.dumps(d).encode()), dtype=torch.uint8)


def dec(t):
    return json.loads(t.numpy().tobytes())


def sd(conf=CONVROT, weight=None, scale=True):
    d = {"l.weight": torch.zeros(4, 4, dtype=torch.int8) if weight is None else weight}
    if scale:
        d["l.weight_scale"] = torch.ones(4)
    if conf is not None:
        d["l.comfy_quant"] = enc(conf)
    return d


@pytest.fixture
def fresh():
    """Module state is per-process; give each test its own install/log state."""
    installed, load_id = m._installed, m._load_id
    m._installed, m._load_id = False, None
    yield
    m._installed, m._load_id = installed, load_id


@pytest.fixture
def ops(fresh):
    """Fake comfy.ops whose original loader just records the state_dict it got."""
    seen = {}

    def spy(module, super_load, state_dict, prefix, *args, **kwargs):
        seen["sd"] = state_dict
        seen["calls"] = seen.get("calls", 0) + 1

    ns = types.SimpleNamespace(_load_quantized_module=spy)
    m._install_into(ns)
    return ns, seen


def load(ns, d):
    return ns._load_quantized_module(object(), None, d, "l.", {}, True, [], [], [])


def test_missing_format_gets_int8_tensorwise(ops):
    ns, seen = ops
    load(ns, sd())
    conf = dec(seen["sd"]["l.comfy_quant"])
    assert conf["format"] == "int8_tensorwise"
    assert conf["convrot"] is True
    assert conf["convrot_groupsize"] == 256
    assert conf["per_row"] is True


def test_existing_format_untouched(ops):
    ns, seen = ops
    d = sd({"format": "float8_e4m3fn"})
    original = d["l.comfy_quant"]
    load(ns, d)
    assert seen["sd"]["l.comfy_quant"] is original


def test_non_int8_weight_untouched(ops):
    ns, seen = ops
    d = sd(weight=torch.zeros(4, 4, dtype=torch.bfloat16))
    original = d["l.comfy_quant"]
    load(ns, d)
    assert seen["sd"]["l.comfy_quant"] is original


def test_missing_weight_scale_untouched(ops):
    ns, seen = ops
    d = sd(scale=False)
    original = d["l.comfy_quant"]
    load(ns, d)
    assert seen["sd"]["l.comfy_quant"] is original


@pytest.mark.parametrize("scale_shape", [(4, 2), (4, 4, 1), (3,)])
def test_non_tensorwise_scale_layout_untouched(ops, scale_shape):
    ns, seen = ops
    d = sd()
    d["l.weight_scale"] = torch.ones(*scale_shape)
    original = d["l.comfy_quant"]
    load(ns, d)
    assert seen["sd"]["l.comfy_quant"] is original


def test_scalar_scale_gets_format(ops):
    ns, seen = ops
    d = sd()
    d["l.weight_scale"] = torch.ones(())
    load(ns, d)
    assert dec(seen["sd"]["l.comfy_quant"])["format"] == "int8_tensorwise"


def test_non_2d_weight_untouched(ops):
    ns, seen = ops
    d = sd(weight=torch.zeros(4, 4, 4, dtype=torch.int8))
    original = d["l.comfy_quant"]
    load(ns, d)
    assert seen["sd"]["l.comfy_quant"] is original


def test_no_comfy_quant_key_untouched(ops):
    ns, seen = ops
    d = sd(conf=None)
    load(ns, d)
    assert "l.comfy_quant" not in seen["sd"]


def test_install_idempotent(ops):
    ns, seen = ops
    assert m._install_into(ns) is True
    assert ns._load_quantized_module._asfp8_wrapped is True
    load(ns, sd())
    assert seen["calls"] == 1


def test_env_off_is_noop(fresh, monkeypatch):
    def spy(module, super_load, state_dict, prefix, *args, **kwargs):
        pass

    fake_comfy = types.SimpleNamespace()
    for off in ("off", "0", "false", "none", "no"):
        fake_ops = types.SimpleNamespace(_load_quantized_module=spy)
        fake_comfy.ops = fake_ops
        monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
        monkeypatch.setitem(sys.modules, "comfy.ops", fake_ops)
        monkeypatch.setenv("ASFP8_QUANT_FORMAT_FIX", off)
        m.install()
        assert not hasattr(fake_ops._load_quantized_module, "_asfp8_wrapped")

    # and on, through the same fake import path, it does wrap
    fake_ops = types.SimpleNamespace(_load_quantized_module=spy)
    fake_comfy.ops = fake_ops
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.ops", fake_ops)
    monkeypatch.delenv("ASFP8_QUANT_FORMAT_FIX", raising=False)
    m.install()
    assert fake_ops._load_quantized_module._asfp8_wrapped is True
