"""ComfyUI prestartup hook for ComfyUI-AppleSilicon-FP8.

Runs BEFORE ComfyUI imports torch — the only point where PyTorch's MPS allocator
watermark ratios can be set (they're read once, at MPS init).

Why this exists
---------------
On Apple Silicon there is no separate VRAM wall — GPU and CPU share one unified
pool. PyTorch's MPS caching allocator is allowed to grow to
``PYTORCH_MPS_HIGH_WATERMARK_RATIO x recommended_max_memory`` before it forces a
purge, and it only *starts* reclaiming cache at
``PYTORCH_MPS_LOW_WATERMARK_RATIO``. The PyTorch defaults are **low=1.4, high=1.7**
— i.e. it won't reclaim until ~1.4x and won't cap until ~1.7x of Apple's
recommended ceiling, both of which EXCEED physical RAM on most Macs. So a heavy
pipeline balloons the *reserved* pool into swap and the OS jetsam-kills the whole
process — even when live tensor memory is small. Measured on a 128 GB M-series:
a SeedVR2 4K run with only ~17 GB live reserved **178 GB** and was killed.

Fix
---
Cap both ratios at/below Apple's recommended_max so the allocator purges its cache
instead of spilling into swap:
  low  = 0.8  -> start reclaiming at 80% of recommended_max
  high = 1.0  -> hard cap at recommended_max
Both scale with the machine (recommended_max is ~80-85% of physical RAM), so this
is a sane default on a 16 GB MacBook and a 128 GB Studio alike.

Overrides
---------
``setdefault`` means an explicit value you set (shell env or ``launchctl setenv``)
always wins. Raise ``high`` if a workflow genuinely needs more and you have the RAM,
or restore PyTorch's defaults with low=1.4 / high=1.7. Disable this hook entirely
with ``APPLESILICON_FP8_MPS_WATERMARK=off``.
"""
import os

if os.environ.get("APPLESILICON_FP8_MPS_WATERMARK", "auto").lower() not in ("off", "0", "false"):
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
    print("[AppleSilicon-FP8/prestartup] MPS allocator watermark capped "
          "(low=%s, high=%s) to keep the reserved pool resident instead of "
          "spilling into swap; set APPLESILICON_FP8_MPS_WATERMARK=off to disable."
          % (os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"],
             os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]), flush=True)
