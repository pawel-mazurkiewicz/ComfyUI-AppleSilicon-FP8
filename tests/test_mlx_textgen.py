from _patches import _mlx_qwen3vl


def test_sampler_kwargs_greedy_when_not_sampling():
    kw = _mlx_qwen3vl._sampler_kwargs(
        do_sample=False, temperature=0.7, top_k=64, top_p=0.95,
        min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0,
    )
    assert kw == {"temperature": 0.0}


def test_sampler_kwargs_maps_all_params_when_sampling():
    kw = _mlx_qwen3vl._sampler_kwargs(
        do_sample=True, temperature=0.7, top_k=64, top_p=0.95,
        min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0,
    )
    assert kw == {
        "temperature": 0.7, "top_p": 0.95, "top_k": 64,
        "min_p": 0.05, "repetition_penalty": 1.05, "presence_penalty": 0.0,
    }
