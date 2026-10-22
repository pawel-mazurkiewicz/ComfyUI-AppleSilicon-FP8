"""MLX backend for Qwen3.5 text generation (internal helper for patch #14).

ComfyUI's Qwen3.5 encoders (MOSS-Image's included) are stock checkpoints, so each size maps
to its stock 4-bit MLX build. ASFP8_MLX_QWEN35_REPO overrides the repo for every size.
"""

import os
import re
import types

from . import _mlx_lm

TAG = "[AppleSilicon-FP8/mlx_qwen35]"

REPOS = {
    "qwen35_08b": "mlx-community/Qwen3.5-0.8B-4bit",
    "qwen35_2b": "mlx-community/Qwen3.5-2B-4bit",
    "qwen35_4b": "mlx-community/Qwen3.5-4B-4bit",
    "qwen35_9b": "mlx-community/Qwen3.5-9B-4bit",
    "qwen35_27b": "mlx-community/Qwen3.5-27B-4bit",
}

# Comfy-Org's plain size/dtype names; fine-tunes and uncensored builds carry extra words
_STOCK = re.compile(r"qwen3\.5_(0\.8b|2b|4b|9b|27b)_(bf16|fp16|fp8_scaled|int8_convrot|w4a8_gptq)\.safetensors")
_logged = set()

available = _mlx_lm.available


def repo_id(model_type):
    return os.environ.get("ASFP8_MLX_QWEN35_REPO") or REPOS.get(model_type)


def route_label():
    return os.environ.get("ASFP8_MLX_QWEN35_REPO") or "mlx-community/Qwen3.5-<size>-4bit"


def generate_text(prompt_text, *, model_type, **kw):
    return _mlx_lm.generate_text(repo_id(model_type), TAG, prompt_text, **kw)


def _log_once(msg):
    if msg not in _logged:
        _logged.add(msg)
        print(msg)


def bind(model_type, files=None):
    """A backend for one size; None when it has no MLX build or the loaded file isn't stock,
    since the stock MLX build would silently stand in for a fine-tune."""
    if repo_id(model_type) is None:
        return None
    if not os.environ.get("ASFP8_MLX_QWEN35_REPO"):
        if files is None:
            _log_once(f"{TAG} loaded Qwen3.5 file unknown (not from a CLIP loader); "
                      f"TextGenerate uses stock {repo_id(model_type)}.")
        elif not all(_STOCK.fullmatch(f) for f in files):
            _log_once(f"{TAG} {', '.join(files)} is not a stock Qwen3.5 file; TextGenerate runs it "
                      "on the eager path (ASFP8_MLX_QWEN35_REPO=<an MLX build of it> keeps MLX).")
            return None
    return types.SimpleNamespace(
        generate_text=lambda prompt_text, **kw: generate_text(prompt_text, model_type=model_type, **kw))
