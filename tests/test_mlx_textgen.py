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


from _patches import mlx_textgen


def test_extract_text_ids_plain_tokens():
    # ComfyUI tokens: {key: [ [ (id, weight), ... ] ]}
    tokens = {"qwen3vl_4b": [[(151644, 1.0), (872, 1.0), (198, 1.0)]]}
    ids, has_media = mlx_textgen._extract_text_ids(tokens)
    assert ids == [151644, 872, 198]
    assert has_media is False


def test_extract_text_ids_detects_media():
    # An image entry is a dict in position [0] instead of an int id.
    tokens = {"qwen3vl_4b": [[(151644, 1.0), ({"type": "image"}, 1.0)]]}
    ids, has_media = mlx_textgen._extract_text_ids(tokens)
    assert has_media is True


class _FakeSub:
    def __init__(self, model_type):
        class _T:  # stand-in for the transformer
            pass
        self.transformer = _T()
        self.transformer.model_type = model_type


class _FakeTok:
    def __init__(self):
        self.tokenizer = object()  # the HF tokenizer sentinel


def test_qwen3vl_hf_tokenizer_found():
    class _CondStage:
        qwen3vl_4b = _FakeSub("qwen3vl_4b")
    class _SD1Tok:
        qwen3vl_4b = _FakeTok()
    hf = mlx_textgen._qwen3vl_hf_tokenizer(_CondStage(), _SD1Tok())
    assert hf is _SD1Tok.qwen3vl_4b.tokenizer


def test_qwen3vl_hf_tokenizer_absent_for_other_models():
    class _CondStage:
        t5xxl = _FakeSub("t5")
    class _SD1Tok:
        t5xxl = _FakeTok()
    hf = mlx_textgen._qwen3vl_hf_tokenizer(_CondStage(), _SD1Tok())
    assert hf is None
