"""Fix: Krea2/LTX2 prompt-expansion TextGenerate is slow on Apple Silicon.

Wraps comfy.sd.CLIP.generate so the autoregressive loop runs under MLX: decode the
already-templated ids to text with ComfyUI's own tokenizer, generate, re-encode. MLX only
ever sees text, so correctness never depends on cross-tokenizer vocab alignment. See
`_ROUTES` for the encoders routed; anything else falls through to the eager generate.
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
    """Identify the MLX backend for a sub-clip by `_modules` key, transformer class name
    or type attribute. Gemma3 keeps model_type on the config class, not the instance."""
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
    """From ComfyUI's tokenize() output, return (batch-0 ids, has_non_int).

    A media entry holds a dict rather than an int, and any non-int means fall back.
    """
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
    """The HF tokenizer for a 'qwen3vl_4b' sub-clip, or None.

    The sub-clip and sub-tokenizer share one attribute key. Iterates _modules rather
    than dir() so no property descriptor fires.
    """
    modules = getattr(cond_stage_model, "_modules", {})
    for key, sub in modules.items():
        transformer = getattr(sub, "transformer", None)
        if getattr(transformer, "model_type", None) == "qwen3vl_4b":
            sub_tok = getattr(sd1_tokenizer, key, None)
            return getattr(sub_tok, "tokenizer", None)
    return None


def _route(cond_stage_model, sd1_tokenizer):
    """(mlx_backend, tokenizer) for the first generatable sub-clip, else (None, None).

    No availability check: install() already gated on a usable backend.
    """
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
    """ids -> templated text."""
    return tok.decode(ids, skip_special_tokens=False)


def _encode_text(tok, text):
    """text -> ids. HF tokenizers have .encode(); comfy's SPieceTokenizer is called."""
    enc = getattr(tok, "encode", None)
    if callable(enc):
        return list(enc(text))
    return list(tok(text)["input_ids"])


def _clip_generate(self, tokens, do_sample=True, max_length=256, temperature=1.0,
                   top_k=50, top_p=0.95, min_p=0.0, repetition_penalty=1.0,
                   seed=None, presence_penalty=0.0, **extra):
    def orig():
        return _orig(self, tokens, do_sample=do_sample, max_length=max_length,
                     temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                     repetition_penalty=repetition_penalty, seed=seed,
                     presence_penalty=presence_penalty, **extra)

    if os.environ.get("ASFP8_DISABLE_MLX_TEXTGEN") == "1":
        return orig()
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
    return orig()


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
