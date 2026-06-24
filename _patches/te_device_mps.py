"""Fix: text encoders always load on CPU on Apple Silicon, even when MPS is free.

ComfyUI picks the text-encoder device in `comfy.model_management.text_encoder_device()`:

    def text_encoder_device():
        if args.gpu_only:
            return get_torch_device()                      # -> mps
        elif vram_state in (HIGH_VRAM, NORMAL_VRAM) or ...:
            ...
        else:
            return torch.device("cpu")                     # <- everything else

On Apple Silicon `vram_state` is *hardcoded* to `VRAMState.SHARED` (unified memory;
see model_management.py "if cpu_state == CPUState.MPS: vram_state = SHARED", which
even overrides --highvram). SHARED is neither HIGH_VRAM nor NORMAL_VRAM, so the
function falls through to `return cpu`. Result: every text encoder runs on CPU
unless you launch with --gpu-only (which also forces offload/VAE/intermediate onto
the GPU and raises peak memory).

For a normal CLIP/T5 encoder that's one forward pass — tolerable. But newer
encoders are autoregressive LLMs (e.g. Krea2, comfy/text_encoders/krea2.py) that
*generate* hundreds of tokens one step at a time. On CPU that's ~1.7s/token —
minutes of wall-clock before sampling even starts.

Fix: on MPS only, wrap text_encoder_device() so that when the stock heuristic
returns CPU we redirect to the Metal device instead — the same load device
--gpu-only would pick, but *scoped to the text encoder*. Offload device is left
alone, so ComfyUI's memory management is otherwise unchanged. text_encoder_dtype
is already fp16 on MPS and these encoders run natively there (the model in the
report loaded as float16, not FP8), so this is purely a placement fix.

Strictly scoped to MPS: on CUDA/CPU boxes get_torch_device() is never an mps
device, so the override never fires and a CUDA box's deliberate CPU offload (real,
separate VRAM) is untouched. An explicit --cpu run keeps get_torch_device() == cpu,
so that is respected too; --gpu-only already returns the GPU, so we no-op there.
"""

import sys

import torch

TAG = "[AppleSilicon-FP8/te_device]"

_mm = None
_orig = None
_installed = False


def _text_encoder_device():
    dev = _orig()
    # Already on a device (e.g. --gpu-only) — nothing to do.
    if dev.type != "cpu":
        return dev
    # Only redirect on Apple Silicon. On CUDA boxes get_torch_device() is "cuda"
    # and the stock CPU choice is a deliberate VRAM-saving offload — leave it. On
    # an explicit --cpu run get_torch_device() is "cpu", so that is respected too.
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
