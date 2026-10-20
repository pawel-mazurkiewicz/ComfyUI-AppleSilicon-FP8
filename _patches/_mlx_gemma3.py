"""MLX backend for Gemma3-12B text generation (internal helper for patch #14).

LTX's encoder is typically an abliterated build, which a stock gemma-3-12b-it would not match,
so that is the default; override with ASFP8_MLX_GEMMA3_REPO.
"""

import os

from . import _mlx_lm

TAG = "[AppleSilicon-FP8/mlx_gemma3]"

DEFAULT_REPO = "mlx-community/gemma-3-12b-it-qat-abliterated-lm-4bit"

available = _mlx_lm.available


def repo_id():
    return os.environ.get("ASFP8_MLX_GEMMA3_REPO", DEFAULT_REPO)


def generate_text(prompt_text, **kw):
    return _mlx_lm.generate_text(repo_id(), TAG, prompt_text, **kw)
