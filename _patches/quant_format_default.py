"""Fix: community INT8 checkpoints whose comfy_quant metadata has no "format" key.

ComfyUI reads `layer_conf.get("format")` and raises "Unknown quantization format" when
it is absent, which rejects otherwise ordinary int8 tensorwise layers (an int8 weight
plus a weight_scale, e.g. Flux2-Klein-9B-True-V3-INT8-ConvRot). The format is filled in
just before comfy's loader reads it; every other key and every other layer is untouched.
"""

import json
import os

import torch

TAG = "[AppleSilicon-FP8/quant_format_default]"

_OFF = ("off", "0", "false", "none", "no")

_installed = False

# comfy hands the same state_dict to every layer of one load, so its identity is the
# load boundary for the once-per-load log line
_load_id = None


def _fixed_conf(state_dict, prefix):
    """Corrected layer config, or None when this layer is not a bare int8 tensorwise one."""
    key = f"{prefix}comfy_quant"
    raw = state_dict.get(key)
    if raw is None:
        return None
    weight = state_dict.get(f"{prefix}weight")
    if getattr(weight, "dtype", None) != torch.int8:
        return None
    if state_dict.get(f"{prefix}weight_scale") is None:
        return None
    try:
        conf = json.loads(raw.numpy().tobytes())
    except Exception:
        return None
    if not isinstance(conf, dict) or "format" in conf:
        return None
    conf["format"] = "int8_tensorwise"
    return conf


def _log_correction(state_dict):
    global _load_id
    if id(state_dict) == _load_id:
        return
    _load_id = id(state_dict)
    print(f"{TAG} comfy_quant has no \"format\": defaulting this load's int8 + "
          f"weight_scale layers to int8_tensorwise.", flush=True)


def _install_into(ops_module) -> bool:
    """Wrap this ops module's quantized loader once. True if wrapped or already wrapped."""
    original = getattr(ops_module, "_load_quantized_module", None)
    if original is None:
        return False
    if getattr(original, "_asfp8_wrapped", False):
        return True

    def _load_quantized_module(module, super_load, state_dict, prefix, *args, **kwargs):
        conf = _fixed_conf(state_dict, prefix)
        if conf is not None:
            _log_correction(state_dict)
            state_dict[f"{prefix}comfy_quant"] = torch.frombuffer(
                bytearray(json.dumps(conf).encode()), dtype=torch.uint8)
        return original(module, super_load, state_dict, prefix, *args, **kwargs)

    _load_quantized_module._asfp8_wrapped = True
    ops_module._load_quantized_module = _load_quantized_module
    return True


def install():
    global _installed
    if _installed:
        return
    if os.environ.get("ASFP8_QUANT_FORMAT_FIX", "").strip().lower() in _OFF:
        return

    try:
        import comfy.ops as ops
    except ImportError:
        return

    if _install_into(ops):
        _installed = True
