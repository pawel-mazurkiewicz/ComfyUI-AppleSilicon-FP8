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


class _FakeHFTok:
    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "TEMPLATED_PROMPT"

    def encode(self, text):
        assert text == "EXPANDED"
        return [10, 11, 12]


class _FakeCLIP:
    """Minimal stand-in exposing what _clip_generate touches."""
    def __init__(self, tokenizer, cond_stage_model):
        self.tokenizer = tokenizer
        self.cond_stage_model = cond_stage_model


def _qwen3vl_clip():
    class _CondStage:
        qwen3vl_4b = _FakeSub("qwen3vl_4b")
    class _SD1Tok:
        qwen3vl_4b = _FakeTok()
    clip = _FakeCLIP(_SD1Tok(), _CondStage())
    # give the discovered tokenizer real encode/decode
    clip.tokenizer.qwen3vl_4b.tokenizer = _FakeHFTok()
    return clip


def test_clip_generate_hijacks_qwen3vl_and_returns_ids(monkeypatch):
    calls = {}

    def fake_generate_text(prompt_text, **kw):
        calls["prompt"] = prompt_text
        calls["kw"] = kw
        return "EXPANDED"

    monkeypatch.setattr(mlx_textgen._mlx_qwen3vl, "generate_text", fake_generate_text)

    clip = _qwen3vl_clip()
    tokens = {"qwen3vl_4b": [[(151644, 1.0), (872, 1.0)]]}
    out = mlx_textgen._clip_generate(
        clip, tokens, do_sample=True, max_length=512, temperature=0.7,
        top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, seed=0,
    )
    assert out == [10, 11, 12]
    assert calls["prompt"] == "TEMPLATED_PROMPT"
    assert calls["kw"]["max_tokens"] == 512


def test_clip_generate_falls_back_when_not_qwen3vl(monkeypatch):
    sentinel = ["ORIG"]
    monkeypatch.setattr(mlx_textgen, "_orig", lambda self, tokens, **kw: sentinel)

    class _CondStage:
        t5xxl = _FakeSub("t5")
    class _SD1Tok:
        t5xxl = _FakeTok()
    clip = _FakeCLIP(_SD1Tok(), _CondStage())
    out = mlx_textgen._clip_generate(clip, {"t5xxl": [[(1, 1.0)]]}, do_sample=True,
                                     max_length=8, temperature=1.0, top_k=0,
                                     top_p=1.0, min_p=0.0, repetition_penalty=1.0)
    assert out is sentinel


def test_clip_generate_falls_back_on_media(monkeypatch):
    sentinel = ["ORIG"]
    monkeypatch.setattr(mlx_textgen, "_orig", lambda self, tokens, **kw: sentinel)
    clip = _qwen3vl_clip()
    tokens = {"qwen3vl_4b": [[(151644, 1.0), ({"type": "image"}, 1.0)]]}
    out = mlx_textgen._clip_generate(clip, tokens, do_sample=True, max_length=8,
                                     temperature=1.0, top_k=0, top_p=1.0,
                                     min_p=0.0, repetition_penalty=1.0)
    assert out is sentinel


def test_clip_generate_falls_back_on_backend_error(monkeypatch):
    sentinel = ["ORIG"]
    monkeypatch.setattr(mlx_textgen, "_orig", lambda self, tokens, **kw: sentinel)

    def boom(prompt_text, **kw):
        raise RuntimeError("mlx exploded")

    monkeypatch.setattr(mlx_textgen._mlx_qwen3vl, "generate_text", boom)
    clip = _qwen3vl_clip()
    tokens = {"qwen3vl_4b": [[(151644, 1.0)]]}
    out = mlx_textgen._clip_generate(clip, tokens, do_sample=True, max_length=8,
                                     temperature=1.0, top_k=0, top_p=1.0,
                                     min_p=0.0, repetition_penalty=1.0)
    assert out is sentinel


def test_clip_generate_disabled_by_env(monkeypatch):
    sentinel = ["ORIG"]
    monkeypatch.setattr(mlx_textgen, "_orig", lambda self, tokens, **kw: sentinel)
    monkeypatch.setenv("ASFP8_DISABLE_MLX_TEXTGEN", "1")
    clip = _qwen3vl_clip()
    out = mlx_textgen._clip_generate(clip, {"qwen3vl_4b": [[(151644, 1.0)]]},
                                     do_sample=True, max_length=8, temperature=1.0,
                                     top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0)
    assert out is sentinel
