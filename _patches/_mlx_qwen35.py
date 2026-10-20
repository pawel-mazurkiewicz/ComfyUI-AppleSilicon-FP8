"""MLX backend for Qwen3.5 text generation (internal helper for patch #14).

ComfyUI's Qwen3.5 encoders (MOSS-Image's included) are stock checkpoints, so each size maps
to its stock 4-bit MLX build. ASFP8_MLX_QWEN35_REPO overrides the repo for every size.
"""

import os
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

available = _mlx_lm.available


def repo_id(model_type):
    return os.environ.get("ASFP8_MLX_QWEN35_REPO") or REPOS.get(model_type)


def route_label():
    return os.environ.get("ASFP8_MLX_QWEN35_REPO") or "mlx-community/Qwen3.5-<size>-4bit"


def generate_text(prompt_text, *, model_type, **kw):
    return _mlx_lm.generate_text(repo_id(model_type), TAG, prompt_text, **kw)


def bind(model_type):
    """A backend for one size, or None when that size has no MLX build."""
    if repo_id(model_type) is None:
        return None
    return types.SimpleNamespace(
        generate_text=lambda prompt_text, **kw: generate_text(prompt_text, model_type=model_type, **kw))
