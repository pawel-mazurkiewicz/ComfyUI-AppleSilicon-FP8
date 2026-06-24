"""MLX backend for Qwen3-VL-4B text generation (internal helper for patch #14).

Isolated from the patch so `mlx` is imported lazily and the patch stays mockable.
Loads an MLX-format Qwen3-VL-4B once and reuses it across calls. Used only for
text-only autoregressive generation (the TextGenerate prompt-expansion step); the
diffusion conditioning encode is untouched.
"""

import importlib.util
import os

TAG = "[AppleSilicon-FP8/mlx_textgen]"

DEFAULT_REPO = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

_MODELS = {}  # repo_id -> (model, processor)


def repo_id():
    return os.environ.get("ASFP8_MLX_QWEN3VL_REPO", DEFAULT_REPO)


def available():
    """True if mlx-vlm can be imported."""
    return importlib.util.find_spec("mlx_vlm") is not None


def _sampler_kwargs(do_sample, temperature, top_k, top_p, min_p,
                    repetition_penalty, presence_penalty):
    # mlx-vlm.generate takes flat sampling kwargs (Task 1 spike: max_tokens,
    # temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty all
    # supported). do_sample=False -> greedy (temp 0; other knobs irrelevant to argmax).
    if not do_sample:
        return {"temperature": 0.0}
    return {
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": float(min_p),
        "repetition_penalty": float(repetition_penalty),
        "presence_penalty": float(presence_penalty),
    }


def _get_model(repo):
    if repo not in _MODELS:
        from mlx_vlm import load  # lazy: only when MLX is actually used
        print(f"{TAG} loading MLX model {repo} (first use; downloads on cache miss).")
        _MODELS[repo] = load(repo)
    return _MODELS[repo]


def generate_text(prompt_text, *, max_tokens, do_sample, temperature, top_k,
                  top_p, min_p, repetition_penalty, presence_penalty=0.0, seed=None):
    """Run text-only autoregressive generation. `prompt_text` is already templated
    (ComfyUI applied the chat template); we pass it through verbatim."""
    import mlx.core as mx
    from mlx_vlm import generate

    model, processor = _get_model(repo_id())
    if seed is not None:
        mx.random.seed(int(seed))

    kwargs = _sampler_kwargs(do_sample, temperature, top_k, top_p, min_p,
                             repetition_penalty, presence_penalty)
    # NOTE (Task 1 spike): pass the pre-formatted prompt positionally; generate() does
    # not re-apply a chat template. Output `.text` is completion-only.
    result = generate(model, processor, prompt_text, max_tokens=int(max_tokens),
                      verbose=False, **kwargs)
    return getattr(result, "text", result)
