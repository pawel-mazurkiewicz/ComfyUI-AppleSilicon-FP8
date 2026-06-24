"""Fix: Krea2's TextGenerate prompt expansion is slow on Apple Silicon.

ComfyUI's `TextGenerate` node (comfy_extras/nodes_textgen.py) calls
`clip.generate(...)` which, for Krea2's Qwen3-VL-4B encoder, runs an eager
autoregressive loop (comfy/text_encoders/llama.py BaseGenerate.generate,
`tqdm("Generating tokens")`, up to max_length=512 full-4B forwards on MPS).
On the reference fp8 run this dominated wall-clock (~50 s of 92 s).

This is autoregressive token generation — exactly what MLX accelerates. We wrap
`comfy.sd.CLIP.generate` so that, on MPS with mlx-vlm installed and a text-only
Qwen3-VL-4B model, we:

  1. decode the incoming (already chat-templated) token ids to text with
     ComfyUI's own HF tokenizer (skip_special_tokens=False),
  2. generate with a cached MLX Qwen3-VL-4B model,
  3. re-encode the output text to token ids with the same HF tokenizer,
  4. return a list[int] — the exact contract BaseGenerate.generate returns, so
     clip.decode(ids) is unchanged.

MLX only ever sees text and ids are produced/consumed by ComfyUI's tokenizer, so
correctness does not depend on cross-tokenizer vocab alignment. Strictly scoped:
non-MPS, non-qwen3vl_4b, multimodal calls, or a missing mlx-vlm all fall through
to the original eager generate. The conditioning encode path (CLIPTextEncode's
12-layer hidden-state tap) is untouched. Disable with ASFP8_DISABLE_MLX_TEXTGEN=1.
"""

import os
import sys

import torch

from _patches import _mlx_qwen3vl

TAG = "[AppleSilicon-FP8/mlx_textgen]"

_orig = None
_installed = False


def _extract_text_ids(tokens):
    """From ComfyUI's tokenize() output {key: [[(id, weight), ...]]} return
    (list[int] ids of batch 0, has_media). A media entry has a dict in slot [0]."""
    batch0 = next(iter(tokens.values()))[0]
    ids = []
    has_media = False
    for entry in batch0:
        elem = entry[0]
        if isinstance(elem, int):
            ids.append(elem)
        else:
            has_media = True
    return ids, has_media


def _qwen3vl_hf_tokenizer(cond_stage_model, sd1_tokenizer):
    """If cond_stage_model carries a sub-clip whose transformer.model_type is
    'qwen3vl_4b', return the matching HF tokenizer (<sub_tokenizer>.tokenizer);
    else None. The sub-clip and sub-tokenizer share the same attribute key
    (SD1ClipModel/SD1Tokenizer both setattr under `name`)."""
    for key in dir(cond_stage_model):
        if key.startswith("_"):
            continue
        sub = getattr(cond_stage_model, key, None)
        transformer = getattr(sub, "transformer", None)
        if getattr(transformer, "model_type", None) == "qwen3vl_4b":
            sub_tok = getattr(sd1_tokenizer, key, None)
            return getattr(sub_tok, "tokenizer", None)
    return None


def install():
    global _orig, _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    if not _mlx_qwen3vl.available():
        return
    try:
        import comfy.sd as sd
    except ImportError:
        return
    if not hasattr(sd, "CLIP") or not hasattr(sd.CLIP, "generate"):
        return

    _orig = sd.CLIP.generate
    sd.CLIP.generate = _clip_generate
    _installed = True
    print(f"{TAG} Qwen3-VL TextGenerate routed through MLX on Apple Silicon "
          f"(repo: {_mlx_qwen3vl.repo_id()}).")
