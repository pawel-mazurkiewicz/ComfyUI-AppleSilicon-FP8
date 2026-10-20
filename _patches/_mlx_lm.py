"""Text-only mlx_lm generation shared by the mlx_textgen backends (one cached model per repo)."""

import importlib.util

from . import _mlx_store

_MODELS = {}  # repo_id -> (model, tokenizer)


def available():
    """True if mlx_lm can be imported."""
    return importlib.util.find_spec("mlx_lm") is not None


def _get_model(repo, tag):
    if repo not in _MODELS:
        from mlx_lm import load  # lazy: only when MLX is actually used
        print(f"{tag} loading MLX model {repo} (first use; downloads on cache miss).")
        _MODELS[repo] = load(_mlx_store.resolve(repo, tag))
    return _MODELS[repo]


def _logits_processors(repetition_penalty, presence_penalty):
    """ComfyUI's eager penalty: each generated token once, never the prompt."""
    rep = float(repetition_penalty or 1.0)
    pres = float(presence_penalty or 0.0)
    if rep == 1.0 and pres == 0.0:
        return None
    import mlx.core as mx

    start = []

    def penalize(tokens, logits):
        if not start:  # first call samples the first token: history is prompt only
            start.append(tokens.size)
        generated = tokens[start[0]:]
        if generated.size == 0:
            return logits
        mask = mx.zeros((logits.shape[-1],), dtype=mx.bool_)
        mask[generated] = True
        pen = mx.where(logits < 0, logits * rep, logits / rep) if rep != 1.0 else logits
        return mx.where(mask, pen - pres, logits)

    return [penalize]


def generate_text(repo, tag, prompt_text, *, max_tokens, do_sample, temperature, top_k,
                  top_p, min_p, repetition_penalty, presence_penalty=0.0, seed=None):
    """`prompt_text` is already chat-templated; mlx_lm.generate does not re-template it."""
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _get_model(repo, tag)
    if seed is not None:
        mx.random.seed(int(seed))

    if do_sample:
        sampler = make_sampler(temp=float(temperature), top_p=float(top_p),
                               top_k=int(top_k), min_p=float(min_p))
    else:
        sampler = make_sampler(temp=0.0)  # greedy

    kwargs = {"max_tokens": int(max_tokens), "sampler": sampler, "verbose": False}
    processors = _logits_processors(repetition_penalty, presence_penalty) if do_sample else None
    if processors:
        kwargs["logits_processors"] = processors
    return generate(model, tokenizer, prompt_text, **kwargs)
