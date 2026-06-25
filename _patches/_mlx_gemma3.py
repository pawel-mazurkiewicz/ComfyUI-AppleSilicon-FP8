"""MLX backend for Gemma3-12B text generation (internal helper for patch #14).

Text-only autoregressive generation via mlx_lm (not mlx_vlm — Gemma3 here is the
LTX2 "Generate Text" prompt-expansion encoder). Loads an MLX-format gemma-3-12b once
and reuses it.

LTX's text encoder is typically an *abliterated* gemma-3-12b (uncensored); a stock
`gemma-3-12b-it` would refuse or behave differently, so the default repo is an
abliterated MLX build. Override with ASFP8_MLX_GEMMA3_REPO to match your exact
variant. The MLX 4-bit weights are a close proxy, not bit-identical to the comfy
checkpoint — fine for prompt enhancement (same contract as the Qwen3-VL route).
"""

import importlib.util
import os

TAG = "[AppleSilicon-FP8/mlx_gemma3]"

# Pre-converted abliterated MLX build (from mlabonne/gemma-3-12b-it-qat-abliterated).
DEFAULT_REPO = "mlx-community/gemma-3-12b-it-qat-abliterated-lm-4bit"

_MODELS = {}  # repo_id -> (model, tokenizer)


def repo_id():
    return os.environ.get("ASFP8_MLX_GEMMA3_REPO", DEFAULT_REPO)


def available():
    """True if mlx_lm can be imported."""
    return importlib.util.find_spec("mlx_lm") is not None


def _get_model(repo):
    if repo not in _MODELS:
        from mlx_lm import load  # lazy: only when MLX is actually used
        print(f"{TAG} loading MLX model {repo} (first use; downloads on cache miss).")
        _MODELS[repo] = load(repo)
    return _MODELS[repo]


def generate_text(prompt_text, *, max_tokens, do_sample, temperature, top_k,
                  top_p, min_p, repetition_penalty, presence_penalty=0.0, seed=None):
    """Run text-only autoregressive generation. `prompt_text` is already chat-templated
    (ComfyUI applied the gemma template); mlx_lm.generate does not re-template, so we
    pass it through verbatim and return the completion text."""
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _get_model(repo_id())
    if seed is not None:
        mx.random.seed(int(seed))

    if do_sample:
        sampler = make_sampler(temp=float(temperature), top_p=float(top_p),
                               top_k=int(top_k), min_p=float(min_p))
    else:
        sampler = make_sampler(temp=0.0)  # greedy

    kwargs = {"max_tokens": int(max_tokens), "sampler": sampler, "verbose": False}
    # repetition_penalty is a logits processor in mlx_lm, not a sampler arg; best-effort.
    try:
        from mlx_lm.sample_utils import make_logits_processors
        if repetition_penalty and float(repetition_penalty) != 1.0:
            kwargs["logits_processors"] = make_logits_processors(
                repetition_penalty=float(repetition_penalty))
    except Exception:
        pass

    return generate(model, tokenizer, prompt_text, **kwargs)
