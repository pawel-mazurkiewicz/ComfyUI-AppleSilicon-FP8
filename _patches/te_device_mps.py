"""Fix: text encoders always load on CPU on Apple Silicon, even when MPS is free.

`vram_state` is hardcoded to SHARED on MPS, which text_encoder_device() treats as
neither HIGH_VRAM nor NORMAL_VRAM and so returns CPU. Only the load device is
redirected; the offload device is left alone.
"""

import sys

import torch

TAG = "[AppleSilicon-FP8/te_device]"

_mm = None
_orig = None
_installed = False


def _text_encoder_device():
    dev = _orig()
    if dev.type != "cpu":   # e.g. --gpu-only already picked a device
        return dev
    # mps only: a CUDA box's CPU choice is a deliberate VRAM offload, and --cpu
    # keeps get_torch_device() == cpu, so both are respected
    td = _mm.get_torch_device()
    if td.type == "mps":
        return td
    return dev


def install():
    global _mm, _orig, _installed
    if _installed:
        return

    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    try:
        import comfy.model_management as mm
    except ImportError:
        return

    if not hasattr(mm, "text_encoder_device"):
        return

    _mm = mm
    _orig = mm.text_encoder_device
    mm.text_encoder_device = _text_encoder_device
    _installed = True
    print(f"{TAG} text_encoder_device redirected CPU->MPS on Apple Silicon (LLM/CLIP encoders run on GPU).")
