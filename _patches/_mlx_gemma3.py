"""MLX backend for Gemma3-12B text generation (internal helper for patch #14).

Text-only generation via mlx_lm, loading one MLX-format model and reusing it. LTX's
encoder is typically an abliterated build, which a stock gemma-3-12b-it would not match,
so that is the default; override with ASFP8_MLX_GEMMA3_REPO.
"""

import importlib.util
import os

TAG = "[AppleSilicon-FP8/mlx_gemma3]"

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
    """Run text-only generation. `prompt_text` is already chat-templated and
    mlx_lm.generate does not re-template, so pass it through verbatim."""
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
    # in mlx_lm this is a logits processor, not a sampler arg
    try:
        from mlx_lm.sample_utils import make_logits_processors
        if repetition_penalty and float(repetition_penalty) != 1.0:
            kwargs["logits_processors"] = make_logits_processors(
                repetition_penalty=float(repetition_penalty))
    except Exception:
        pass

    return generate(model, tokenizer, prompt_text, **kwargs)
