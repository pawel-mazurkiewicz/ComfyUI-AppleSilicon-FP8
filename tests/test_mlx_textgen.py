import os

import pytest

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


from _patches import mlx_textgen  # noqa: E402


def test_extract_text_ids_plain_tokens():
    # ComfyUI tokens: {key: [ [ (id, weight), ... ] ]}
    tokens = {"qwen3vl_4b": [[(151644, 1.0), (872, 1.0), (198, 1.0)]]}
    ids, has_non_int = mlx_textgen._extract_text_ids(tokens)
    assert ids == [151644, 872, 198]
    assert has_non_int is False


def test_extract_text_ids_detects_media():
    # An image entry is a dict in position [0] instead of an int id.
    tokens = {"qwen3vl_4b": [[(151644, 1.0), ({"type": "image"}, 1.0)]]}
    ids, has_non_int = mlx_textgen._extract_text_ids(tokens)
    assert has_non_int is True


class _FakeSub:
    def __init__(self, model_type):
        class _T:  # stand-in for the transformer
            pass
        self.transformer = _T()
        self.transformer.model_type = model_type


class _FakeTok:
    def __init__(self):
        self.tokenizer = object()  # the HF tokenizer sentinel


def _fake_cond_stage(**named_subs):
    """Build a minimal cond_stage_model fake with a _modules dict (mimics nn.Module)."""
    obj = object.__new__(type("_FakeCondStage", (), {}))
    obj._modules = named_subs
    return obj


def _fake_sd1_tok(**named_toks):
    """Build a minimal SD1Tokenizer fake with named tokenizer sub-objects as attributes."""
    obj = object.__new__(type("_FakeSD1Tok", (), {}))
    for k, v in named_toks.items():
        setattr(obj, k, v)
    return obj


def test_qwen3vl_hf_tokenizer_found():
    sub = _FakeSub("qwen3vl_4b")
    tok = _FakeTok()
    cond = _fake_cond_stage(qwen3vl_4b=sub)
    sd1 = _fake_sd1_tok(qwen3vl_4b=tok)
    hf = mlx_textgen._qwen3vl_hf_tokenizer(cond, sd1)
    assert hf is tok.tokenizer


def test_qwen3vl_hf_tokenizer_absent_for_other_models():
    cond = _fake_cond_stage(t5xxl=_FakeSub("t5"))
    sd1 = _fake_sd1_tok(t5xxl=_FakeTok())
    hf = mlx_textgen._qwen3vl_hf_tokenizer(cond, sd1)
    assert hf is None


class _FakeHFTok:
    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "TEMPLATED_PROMPT"

    def encode(self, text):
        return [10, 11, 12]


class _FakeCLIP:
    """Minimal stand-in exposing what _clip_generate touches."""
    def __init__(self, tokenizer, cond_stage_model):
        self.tokenizer = tokenizer
        self.cond_stage_model = cond_stage_model


def _qwen3vl_clip():
    sub_tok = _FakeTok()
    sub_tok.tokenizer = _FakeHFTok()
    cond = _fake_cond_stage(qwen3vl_4b=_FakeSub("qwen3vl_4b"))
    sd1 = _fake_sd1_tok(qwen3vl_4b=sub_tok)
    return _FakeCLIP(sd1, cond)


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
    assert isinstance(out, list) and all(isinstance(x, int) for x in out)
    assert out == [10, 11, 12]
    assert calls["prompt"] == "TEMPLATED_PROMPT"
    assert calls["kw"]["max_tokens"] == 512


def test_clip_generate_falls_back_when_not_qwen3vl(monkeypatch):
    sentinel = ["ORIG"]
    monkeypatch.setattr(mlx_textgen, "_orig", lambda self, tokens, **kw: sentinel)

    cond = _fake_cond_stage(t5xxl=_FakeSub("t5"))
    sd1 = _fake_sd1_tok(t5xxl=_FakeTok())
    clip = _FakeCLIP(sd1, cond)
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


# --- gemma3 (LTX2) routing + SentencePiece tokenizer adapter ---


class _FakeGemmaSub:
    def __init__(self):
        class _T:
            transformer_type = "gemma3"   # gemma3 reports transformer_type, not model_type
        self.transformer = _T()


class _FakeSPieceTok:
    """comfy SentencePiece tokenizer: decode(ids, skip_special_tokens=), no .encode,
    encode via __call__ -> {'input_ids': [...]}."""
    def decode(self, ids, skip_special_tokens=False):
        return "GEMMA_TEMPLATED"
    def __call__(self, text):
        return {"input_ids": [7, 8, 9]}


def test_route_gemma3():
    sub_tok = _FakeTok()
    sub_tok.tokenizer = _FakeSPieceTok()
    cond = _fake_cond_stage(gemma3_12b=_FakeGemmaSub())
    sd1 = _fake_sd1_tok(gemma3_12b=sub_tok)
    backend, tok = mlx_textgen._route(cond, sd1)
    assert backend is mlx_textgen._mlx_gemma3
    assert isinstance(tok, _FakeSPieceTok)


def test_route_none_for_unknown():
    cond = _fake_cond_stage(t5xxl=_FakeSub("t5"))
    sd1 = _fake_sd1_tok(t5xxl=_FakeTok())
    backend, tok = mlx_textgen._route(cond, sd1)
    assert backend is None and tok is None


def test_encode_text_hf_uses_encode():
    assert mlx_textgen._encode_text(_FakeHFTok(), "x") == [10, 11, 12]


def test_encode_text_spiece_uses_call():
    assert mlx_textgen._encode_text(_FakeSPieceTok(), "x") == [7, 8, 9]


def test_clip_generate_routes_gemma3_via_mlx_lm(monkeypatch):
    calls = {}

    def fake_gemma_gen(prompt_text, **kw):
        calls["prompt"] = prompt_text
        return "ENHANCED PROMPT"

    monkeypatch.setattr(mlx_textgen._mlx_gemma3, "generate_text", fake_gemma_gen)

    sub_tok = _FakeTok()
    sub_tok.tokenizer = _FakeSPieceTok()
    cond = _fake_cond_stage(gemma3_12b=_FakeGemmaSub())
    sd1 = _fake_sd1_tok(gemma3_12b=sub_tok)
    clip = _FakeCLIP(sd1, cond)
    out = mlx_textgen._clip_generate(
        clip, {"gemma3_12b": [[(2, 1.0), (105, 1.0)]]}, do_sample=True, max_length=512,
        temperature=0.7, top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, seed=0,
    )
    assert out == [7, 8, 9]                 # encoded via SPiece __call__
    assert calls["prompt"] == "GEMMA_TEMPLATED"


_run_integration = os.environ.get("ASFP8_RUN_MLX_INTEGRATION") == "1"
requires_mlx_integration = pytest.mark.skipif(
    not _run_integration,
    reason="set ASFP8_RUN_MLX_INTEGRATION=1 to run (downloads ~2.2 GB and uses MPS)",
)


@requires_mlx_integration
def test_mlx_generate_text_real():
    from _patches import _mlx_qwen3vl
    assert _mlx_qwen3vl.available()
    pre = "<|im_start|>user\nName three primary colors.<|im_end|>\n<|im_start|>assistant\n"
    out = _mlx_qwen3vl.generate_text(
        pre, max_tokens=40, do_sample=False, temperature=0.0, top_k=0,
        top_p=1.0, min_p=0.0, repetition_penalty=1.0, seed=0,
    )
    assert isinstance(out, str) and len(out.strip()) > 0
