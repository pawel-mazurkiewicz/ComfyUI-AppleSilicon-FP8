"""ComfyUI prestartup hook: cap the MPS allocator watermarks before torch is imported.

torch's defaults (low=1.4, high=1.7 x recommended_max) exceed physical RAM on unified
memory, so the reserved pool spills into swap and the OS jetsam-kills the process.
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
