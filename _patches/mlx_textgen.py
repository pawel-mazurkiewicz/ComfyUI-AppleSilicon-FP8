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

from . import _mlx_qwen3vl

TAG = "[AppleSilicon-FP8/mlx_textgen]"

_orig = None
_installed = False


class _Fallback(Exception):
    """Internal: signal 'not our case, use the eager path' without logging."""


def _extract_text_ids(tokens):
    """From ComfyUI's tokenize() output {key: [[(id, weight), ...]]} return
    (list[int] ids of batch 0, has_non_int). A media/image entry has a dict in
    slot [0] (placed by Qwen3VLTokenizer.tokenize_with_weights); any other
    non-int entry is also treated as non-text and triggers fallback."""
    batch0 = next(iter(tokens.values()))[0]
    ids = []
    has_non_int = False
    for entry in batch0:
        elem = entry[0]
        if isinstance(elem, int):
            ids.append(elem)
        else:
            has_non_int = True
    return ids, has_non_int


def _qwen3vl_hf_tokenizer(cond_stage_model, sd1_tokenizer):
    """If cond_stage_model carries a sub-clip whose transformer.model_type is
    'qwen3vl_4b', return the matching HF tokenizer (<sub_tokenizer>.tokenizer);
    else None. The sub-clip and sub-tokenizer share the same attribute key
    (SD1ClipModel/SD1Tokenizer both setattr under `name`).

    Iterates _modules (nn.Module's registered submodule ordered dict) to avoid
    triggering property descriptors and to stay O(submodules) rather than O(dir())."""
    modules = getattr(cond_stage_model, "_modules", {})
    for key, sub in modules.items():
        transformer = getattr(sub, "transformer", None)
        if getattr(transformer, "model_type", None) == "qwen3vl_4b":
            sub_tok = getattr(sd1_tokenizer, key, None)
            return getattr(sub_tok, "tokenizer", None)
    return None


def _clip_generate(self, tokens, do_sample=True, max_length=256, temperature=1.0,
                   top_k=50, top_p=0.95, min_p=0.0, repetition_penalty=1.0,
                   seed=None, presence_penalty=0.0):
    if os.environ.get("ASFP8_DISABLE_MLX_TEXTGEN") == "1":
        return _orig(self, tokens, do_sample=do_sample, max_length=max_length,
                     temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                     repetition_penalty=repetition_penalty, seed=seed,
                     presence_penalty=presence_penalty)
    try:
        hf_tok = _qwen3vl_hf_tokenizer(self.cond_stage_model, self.tokenizer)
        if hf_tok is None:
            raise _Fallback
        ids, has_non_int = _extract_text_ids(tokens)
        if has_non_int or not ids:
            raise _Fallback

        prompt_text = hf_tok.decode(ids, skip_special_tokens=False)
        out_text = _mlx_qwen3vl.generate_text(
            prompt_text, max_tokens=max_length, do_sample=do_sample,
            temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
            repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
            seed=seed,
        )
        return list(hf_tok.encode(out_text))
    except _Fallback:
        pass
    except Exception as e:  # never break a render: fall back to the eager path
        print(f"{TAG} MLX generation failed ({e!r}); falling back to eager.")
    return _orig(self, tokens, do_sample=do_sample, max_length=max_length,
                 temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                 repetition_penalty=repetition_penalty, seed=seed,
                 presence_penalty=presence_penalty)


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
