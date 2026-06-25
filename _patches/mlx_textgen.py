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
correctness does not depend on cross-tokenizer vocab alignment.

Two generatable encoders are routed (see `_ROUTES`): Krea2's Qwen3-VL-4B (via mlx-vlm)
and LTX2's Gemma3-12B prompt encoder (via mlx-lm; the "Generate Text" node, otherwise
~20 s/token eager). Each maps to an MLX repo (`ASFP8_MLX_QWEN3VL_REPO` /
`ASFP8_MLX_GEMMA3_REPO`). Strictly scoped: non-MPS, an unrecognised encoder, multimodal
calls, or a missing MLX package all fall through to the original eager generate. The
conditioning encode path (CLIPTextEncode's hidden-state tap) is untouched. Disable with
ASFP8_DISABLE_MLX_TEXTGEN=1.
"""

import os
import sys

import torch

from . import _mlx_gemma3, _mlx_qwen3vl

TAG = "[AppleSilicon-FP8/mlx_textgen]"

_orig = None
_installed = False

_logged_miss = False


def _backend_for(key, sub):
    """Identify the MLX backend for a sub-clip, robust to comfy internals: match on the
    `_modules` key (the model's attribute name), the transformer instance's class name,
    or a type attribute if present. Gemma3's transformer exposes none of model_type/
    transformer_type on the instance (they live on the config class), so key/class name
    are the reliable signals."""
    tr = getattr(sub, "transformer", None)
    cls = type(tr).__name__ if tr is not None else ""
    mt = getattr(tr, "model_type", None)
    tt = getattr(tr, "transformer_type", None)
    if key == "qwen3vl_4b" or mt == "qwen3vl_4b" or "Qwen3VL" in cls:
        return _mlx_qwen3vl
    if key == "gemma3_12b" or tt == "gemma3" or cls.startswith("Gemma3"):
        return _mlx_gemma3
    return None


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


def _route(cond_stage_model, sd1_tokenizer):
    """Find the first sub-clip whose transformer matches a known generatable model and
    return (mlx_backend, tokenizer) for it, else (None, None). The sub-clip and its
    tokenizer share the same attribute key. No availability check here — install()
    already gated on a usable backend, and _clip_generate falls back on any error."""
    global _logged_miss
    modules = getattr(cond_stage_model, "_modules", {})
    for key, sub in modules.items():
        backend = _backend_for(key, sub)
        if backend is None:
            continue
        sub_tok = getattr(sd1_tokenizer, key, None)
        tok = getattr(sub_tok, "tokenizer", None)
        if tok is not None:
            return backend, tok
    if not _logged_miss:
        _logged_miss = True
        seen = {k: type(getattr(s, "transformer", None)).__name__ for k, s in modules.items()}
        print(f"{TAG} no MLX route matched (eager fallback); sub-models seen: {seen}")
    return None, None


def _decode_ids(tok, ids):
    """ids -> templated text. HF and comfy's SentencePiece tokenizer both expose
    decode(ids, skip_special_tokens=...)."""
    return tok.decode(ids, skip_special_tokens=False)


def _encode_text(tok, text):
    """text -> ids. HF tokenizers use .encode(); comfy's SPieceTokenizer has no encode
    and is called as tok(text) -> {'input_ids': [...]}."""
    enc = getattr(tok, "encode", None)
    if callable(enc):
        return list(enc(text))
    return list(tok(text)["input_ids"])


def _clip_generate(self, tokens, do_sample=True, max_length=256, temperature=1.0,
                   top_k=50, top_p=0.95, min_p=0.0, repetition_penalty=1.0,
                   seed=None, presence_penalty=0.0):
    if os.environ.get("ASFP8_DISABLE_MLX_TEXTGEN") == "1":
        return _orig(self, tokens, do_sample=do_sample, max_length=max_length,
                     temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                     repetition_penalty=repetition_penalty, seed=seed,
                     presence_penalty=presence_penalty)
    try:
        backend, tok = _route(self.cond_stage_model, self.tokenizer)
        if backend is None:
            raise _Fallback
        ids, has_non_int = _extract_text_ids(tokens)
        if has_non_int or not ids:
            raise _Fallback

        prompt_text = _decode_ids(tok, ids)
        out_text = backend.generate_text(
            prompt_text, max_tokens=max_length, do_sample=do_sample,
            temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
            repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
            seed=seed,
        )
        return _encode_text(tok, out_text)
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
    if not (_mlx_qwen3vl.available() or _mlx_gemma3.available()):
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
    routes = []
    if _mlx_qwen3vl.available():
        routes.append(f"qwen3vl_4b -> {_mlx_qwen3vl.repo_id()}")
    if _mlx_gemma3.available():
        routes.append(f"gemma3_12b -> {_mlx_gemma3.repo_id()}")
    print(f"{TAG} TextGenerate routed through MLX on Apple Silicon ({'; '.join(routes)}).")
