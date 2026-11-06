import pytest
import torch

from _patches import qk_norm_rope_mps as m
from _patches import rope_fast_mps

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
DEV = "mps"
EPS = 1e-5
ZIMAGE = (1, 4160, 30, 128)          # B, S, H, D at 1024^2
SMALL = (2, 37, 3, 16)


@pytest.fixture
def installed():
    m.install_for_test()
    m._backend_events.clear()
    try:
        yield
    finally:
        m.uninstall_for_test()


def _table(B, S, D, bshd=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(B, S, D // 2, generator=g) * 3
    c, s = th.cos(), th.sin()
    t = torch.stack([c, -s, s, c], -1)
    t = t.view(B, S, 1, D // 2, 2, 2) if bshd else t.view(B, 1, S, D // 2, 2, 2)
    return t.to(DEV)


def _qkv_views(B, S, H, D, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    qkv = (torch.randn(B, S, 3 * H * D, generator=g) * 3).to(dtype).to(DEV)
    q, k, _ = (t.view(B, S, H, D) for t in torch.split(qkv, [H * D] * 3, dim=-1))
    return q, k


def _weight(D, dtype, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(D, generator=g) + 0.5).to(dtype).to(DEV)


def _ref32(x, w, f):
    xf = x.float()
    n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
    xr = n.reshape(*n.shape[:-1], -1, 1, 2)
    return (f[..., 0] * xr[..., 0] + f[..., 1] * xr[..., 1]).reshape(n.shape)


def _rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm()).item()


def _eager(name, *a):
    return m._orig[name](*a)


# ---------------------------------------------------------------------------
# kernel vs real eager, accuracy against fp32
# ---------------------------------------------------------------------------
@mps
@pytest.mark.parametrize("shape", [ZIMAGE, SMALL, (1, 5, 2, 6), (1, 16, 2, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_rms_rope_matches_eager_and_fp32(installed, shape, dtype):
    B, S, H, D = shape
    q, k = _qkv_views(B, S, H, D, dtype)
    w, f = _weight(D, dtype), _table(1, S, D)
    eq, ek = _eager("rms_rope", q, k, f, w, w, EPS)
    m._backend_events.clear()
    fq, fk = m.rms_rope_fused(q, k, f, w, w, EPS)
    torch.mps.synchronize()
    assert [e[1] for e in m._backend_events] == ["kernel", "kernel"]
    for x, e, o in ((q, eq, fq), (k, ek, fk)):
        assert o.shape == e.shape and o.dtype == e.dtype
        assert torch.allclose(o.contiguous().float(), e.contiguous().float(),
                              atol=0.1 if dtype != torch.float32 else 1e-4,
                              rtol=0.05 if dtype != torch.float32 else 1e-4)
        r = _ref32(x, w, f)
        if dtype == torch.float32:
            assert _rel(o, r) < 1e-5
        else:
            assert _rel(o, r) <= 1.05 * _rel(e, r)


@mps
def test_rms_rope1_matches_eager_zimage(installed):
    B, S, H, D = ZIMAGE
    q, _ = _qkv_views(B, S, H, D, torch.bfloat16)
    w, f = _weight(D, torch.bfloat16), _table(1, S, D)
    e = _eager("rms_rope1", q, f, w, EPS)
    m._backend_events.clear()
    o = m.rms_rope1_fused(q, f, w, EPS)
    assert m._last_backend() == "kernel"
    r = _ref32(q, w, f)
    assert _rel(o, r) <= 1.05 * _rel(e, r)
    assert _rel(o.contiguous(), e.contiguous()) < 1e-2


@mps
def test_public_comfy_kitchen_entry_reaches_the_kernel(installed):
    import comfy_kitchen as ck
    B, S, H, D = SMALL
    q, k = _qkv_views(B, S, H, D, torch.bfloat16)
    w, f = _weight(D, torch.bfloat16), _table(1, S, D)
    m._backend_events.clear()
    ck.rms_rope(q, k, f, w, w, EPS)
    ck.rms_rope1(q, f, w, EPS)
    assert [e[1] for e in m._backend_events] == ["kernel"] * 3


@mps
def test_k_scale_defaults_to_q_scale(installed):
    B, S, H, D = SMALL
    q, k = _qkv_views(B, S, H, D, torch.bfloat16)
    w, f = _weight(D, torch.bfloat16), _table(1, S, D)
    a = m.rms_rope_fused(q, k, f, w)
    b = m.rms_rope_fused(q, k, f, w, w)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ---------------------------------------------------------------------------
# layouts
# ---------------------------------------------------------------------------
@mps
@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("batch_table", [False, True])
def test_bshd_strided_views_batch_and_tables(installed, B, batch_table):
    S, H, D = 37, 3, 16
    q, k = _qkv_views(B, S, H, D, torch.bfloat16)
    w = _weight(D, torch.bfloat16)
    f = _table(B if batch_table else 1, S, D)
    for x in (q, k):                       # k carries a storage offset
        e = _eager("rms_rope1", x, f, w, EPS)
        m._backend_events.clear()
        o = m.rms_rope1_fused(x, f, w, EPS)
        assert m._last_backend() == "kernel"
        assert _rel(o.contiguous(), e.contiguous()) < 1e-2
        assert _rel(o, _ref32(x, w, f)) <= 1.05 * _rel(e, _ref32(x, w, f))


@mps
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("batch_table", [False, True])
def test_bhsd_input(installed, B, batch_table):
    H, S, D = 4, 29, 32
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(B, H, S, D, generator=g) * 2).to(torch.bfloat16).to(DEV)
    w = _weight(D, torch.bfloat16)
    f = _table(B if batch_table else 1, S, D, bshd=False)
    e = _eager("rms_rope1", x, f, w, EPS)
    m._backend_events.clear()
    o = m.rms_rope1_fused(x, f, w, EPS)
    assert m._last_backend() == "kernel"
    assert o.shape == x.shape
    assert _rel(o.contiguous(), e.contiguous()) < 1e-2
    assert _rel(o, _ref32(x, w, f)) <= 1.05 * _rel(e, _ref32(x, w, f))


@mps
def test_bhsd_input_that_is_a_permuted_view(installed):
    B, S, H, D = 2, 21, 4, 16
    q, _ = _qkv_views(B, S, H, D, torch.bfloat16)
    x = q.movedim(1, 2)                    # [B,H,S,D] view, strides not contiguous
    w = _weight(D, torch.bfloat16)
    f = _table(1, S, D, bshd=False)
    e = _eager("rms_rope1", x, f, w, EPS)
    o = m.rms_rope1_fused(x, f, w, EPS)
    assert m._last_backend() == "kernel"
    assert _rel(o.contiguous(), e.contiguous()) < 1e-2


@mps
def test_odd_head_dim_takes_the_original(installed, monkeypatch):
    sentinel = torch.zeros(1, device=DEV)
    monkeypatch.setitem(m._orig, "rms_rope1", lambda *a: sentinel)
    x = torch.randn(1, 5, 2, 15, device=DEV).to(torch.bfloat16)
    w = _weight(15, torch.bfloat16)
    f = torch.zeros(1, 5, 1, 7, 2, 2, device=DEV)
    assert m.rms_rope1_fused(x, f, w, EPS) is sentinel
    assert m._last_backend() == "fallback"


# ---------------------------------------------------------------------------
# output contract: same shape, dtype, strides and viewability as eager
# ---------------------------------------------------------------------------
@mps
@pytest.mark.parametrize("with_rope_fast", [True, False])
@pytest.mark.parametrize("bshd", [True, False])
def test_output_strides_match_eager(with_rope_fast, bshd):
    if with_rope_fast:
        rope_fast_mps.install_for_test()
    m.install_for_test()
    try:
        B, S, H, D = SMALL
        if bshd:
            x, _ = _qkv_views(B, S, H, D, torch.bfloat16)
        else:
            x = torch.randn(B, H, S, D, device=DEV).to(torch.bfloat16)
        w, f = _weight(D, torch.bfloat16), _table(1, S, D, bshd=bshd)
        e = _eager("rms_rope1", x, f, w, EPS)
        m._backend_events.clear()
        o = m.rms_rope1_fused(x, f, w, EPS)
        assert m._last_backend() == "kernel"
        assert (o.shape, o.dtype, o.stride(), o.storage_offset(), o.is_contiguous()) == \
               (e.shape, e.dtype, e.stride(), e.storage_offset(), e.is_contiguous())
        assert o.view(-1).numel() == e.view(-1).numel()          # callers may .view()
        if bshd:
            assert o.view(B, S, H * D).shape == e.view(B, S, H * D).shape
    finally:
        m.uninstall_for_test()
        if with_rope_fast:
            rope_fast_mps.uninstall_for_test()


# ---------------------------------------------------------------------------
# per-call fallbacks (sentinel eager proves the original ran)
# ---------------------------------------------------------------------------
def _case(B=2, S=37, H=3, D=16, dtype=torch.bfloat16):
    q, _ = _qkv_views(B, S, H, D, dtype)
    return q, _table(1, S, D), _weight(D, dtype)


@pytest.fixture
def sentinel(monkeypatch):
    s = torch.zeros(1)
    calls = []
    monkeypatch.setitem(m._orig, "rms_rope1", lambda *a: calls.append(a) or s)
    s.calls = calls
    return s


@mps
def test_dtype_the_kernel_does_not_cover_falls_back(installed, sentinel, monkeypatch):
    q, f, w = _case()
    monkeypatch.delitem(m._MSL_T, torch.bfloat16)
    assert m.rms_rope1_fused(q, f, w, EPS) is sentinel
    assert m._last_backend() == "fallback"


@mps
def test_real_unsupported_dtype_returns_the_eager_result(installed):
    x = torch.randn(1, 6, 2, 8, dtype=torch.float64)
    w = torch.rand(8, dtype=torch.float64) + 0.5
    f = _table(1, 6, 8).cpu()
    e = _eager("rms_rope1", x, f, w, EPS)
    o = m.rms_rope1_fused(x, f, w, EPS)
    assert m._last_backend() == "fallback" and torch.equal(o, e)


@mps
def test_scale_dtype_mismatch_falls_back(installed, sentinel):
    q, f, _ = _case()
    assert m.rms_rope1_fused(q, f, _weight(16, torch.float32), EPS) is sentinel


@mps
def test_non_fp32_table_falls_back(installed, sentinel):
    q, f, w = _case()
    assert m.rms_rope1_fused(q, f.to(torch.bfloat16), w, EPS) is sentinel


@mps
def test_table_length_mismatch_falls_back(installed, sentinel):
    q, _, w = _case()
    assert m.rms_rope1_fused(q, _table(1, 40, 16), w, EPS) is sentinel


@mps
def test_ambiguous_single_token_falls_back(installed, sentinel):
    q, _, w = _case(S=1)
    assert m.rms_rope1_fused(q, _table(1, 1, 16), w, EPS) is sentinel


@mps
def test_cpu_tensor_returns_the_eager_result(installed):
    x = torch.randn(1, 6, 2, 8).to(torch.bfloat16)
    w = (torch.rand(8) + 0.5).to(torch.bfloat16)
    f = _table(1, 6, 8).cpu()
    e = _eager("rms_rope1", x, f, w, EPS)
    o = m.rms_rope1_fused(x, f, w, EPS)
    assert m._last_backend() == "fallback" and torch.equal(o, e)


@mps
def test_table_on_another_device_falls_back(installed, sentinel):
    q, f, w = _case()
    assert m.rms_rope1_fused(q, f.cpu(), w, EPS) is sentinel


@mps
def test_grad_tracking_falls_back(installed, sentinel):
    q, f, w = _case()
    q = q.detach().float().requires_grad_(True).to(torch.float32)
    assert m.rms_rope1_fused(q, f, w.float(), EPS) is sentinel


@mps
def test_torch_compile_falls_back(installed, sentinel, monkeypatch):
    q, f, w = _case()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert m.rms_rope1_fused(q, f, w, EPS) is sentinel


@mps
def test_grid_limit_falls_back_and_boundary_still_runs(installed, monkeypatch):
    B, S = 2, 37
    q, f, w = _case(B=B, S=S)
    monkeypatch.setattr(m, "_MAXZ", B * S)
    e = _eager("rms_rope1", q, f, w, EPS)
    o = m.rms_rope1_fused(q, f, w, EPS)
    assert m._last_backend() == "kernel"
    assert _rel(o, e) < 1e-2
    monkeypatch.setattr(m, "_MAXZ", B * S - 1)
    o = m.rms_rope1_fused(q, f, w, EPS)
    assert m._last_backend() == "fallback" and torch.equal(o, e)


@mps
def test_head_count_over_grid_limit_falls_back(installed, monkeypatch):
    q, f, w = _case(H=3)
    monkeypatch.setattr(m, "_MAXZ", 2)
    m.rms_rope1_fused(q, f, w, EPS)
    assert m._last_backend() == "fallback"


# ---------------------------------------------------------------------------
# self-check / latch
# ---------------------------------------------------------------------------
def _tag_lines(capsys):
    return [ln for ln in capsys.readouterr().out.splitlines() if m.TAG in ln]


@mps
def test_self_check_failure_latches_off_with_one_line(installed, monkeypatch, capsys):
    real, launches = m._launch, []

    def wrong(*a, **k):
        launches.append(1)
        return real(*a, **k) * 2

    monkeypatch.setattr(m, "_launch", wrong)
    q, f, w = _case()
    e = _eager("rms_rope1", q, f, w, EPS)
    for _ in range(3):
        o = m.rms_rope1_fused(q, f, w, EPS)
        assert torch.equal(o, e)
        assert m._last_backend() == "fallback"
    assert m._checked[torch.bfloat16] is False
    n = len(launches)
    assert n >= 1
    m.rms_rope1_fused(q, f, w, EPS)
    assert len(launches) == n                      # never retried
    assert len(_tag_lines(capsys)) == 1


@mps
def test_self_check_exception_latches_off_with_one_line(installed, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(m, "_launch", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(RuntimeError("boom")))
    q, f, w = _case()
    e = _eager("rms_rope1", q, f, w, EPS)
    for _ in range(3):
        assert torch.equal(m.rms_rope1_fused(q, f, w, EPS), e)
    assert len(calls) == 1
    assert len(_tag_lines(capsys)) == 1


@mps
def test_self_check_is_per_dtype(installed, monkeypatch, capsys):
    real = m._launch

    def wrong_for_bf16(x, *a, **k):
        o = real(x, *a, **k)
        return o * 2 if x.dtype == torch.bfloat16 else o

    monkeypatch.setattr(m, "_launch", wrong_for_bf16)
    qb, fb, wb = _case(dtype=torch.bfloat16)
    qh, fh, wh = _case(dtype=torch.float16)
    m.rms_rope1_fused(qb, fb, wb, EPS)
    assert m._last_backend() == "fallback"
    m.rms_rope1_fused(qh, fh, wh, EPS)
    assert m._last_backend() == "kernel"
    assert m._checked == {torch.bfloat16: False, torch.float16: True}


@mps
def test_runtime_kernel_error_falls_back_then_latches(installed, monkeypatch, capsys):
    q, f, w = _case()
    e = _eager("rms_rope1", q, f, w, EPS)
    m.rms_rope1_fused(q, f, w, EPS)                # verify the dtype first
    assert m._last_backend() == "kernel"
    calls = []
    monkeypatch.setattr(m, "_launch", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(RuntimeError("boom")))
    capsys.readouterr()
    for _ in range(3):
        assert torch.equal(m.rms_rope1_fused(q, f, w, EPS), e)
    assert len(calls) == 1
    assert len(_tag_lines(capsys)) == 1


@mps
def test_self_check_runs_once_per_dtype(installed, monkeypatch):
    real, n = m._verify, []
    monkeypatch.setattr(m, "_verify", lambda *a: n.append(1) or real(*a))
    q, f, w = _case()
    for _ in range(3):
        m.rms_rope1_fused(q, f, w, EPS)
    assert len(n) == 1


# ---------------------------------------------------------------------------
# install gating
# ---------------------------------------------------------------------------
def _snapshot():
    import comfy_kitchen.backends.eager as e
    import comfy_kitchen.backends.eager.rope as r
    return [getattr(mod, n) for mod in (e, r) for n in ("rms_rope", "rms_rope1")]


@pytest.mark.parametrize("value", ["off", "0", "false", "none", "no", "OFF", " off "])
def test_env_off_leaves_comfy_kitchen_untouched(monkeypatch, value, capsys):
    before = _snapshot()
    monkeypatch.setenv("ASFP8_QK_NORM_ROPE", value)
    m.install()
    assert _snapshot() == before
    assert not m._installed
    assert m.TAG not in capsys.readouterr().out


@mps
def test_env_on_installs_and_uninstall_restores(monkeypatch, capsys):
    before = _snapshot()
    monkeypatch.setenv("ASFP8_QK_NORM_ROPE", "1")
    try:
        m.install()
        assert m._installed
        assert all(a is not b for a, b in zip(_snapshot(), before))
        assert "active" in capsys.readouterr().out
    finally:
        m.uninstall_for_test()
    assert _snapshot() == before


def test_no_compile_shader_does_not_install(monkeypatch):
    from _patches import _caps
    before = _snapshot()
    monkeypatch.delenv("ASFP8_QK_NORM_ROPE", raising=False)
    monkeypatch.setattr(_caps, "has_compile_shader", lambda: False)
    m.install()
    assert _snapshot() == before and not m._installed


def test_install_failure_is_not_fatal_and_not_claimed(monkeypatch, capsys):
    monkeypatch.setenv("ASFP8_QK_NORM_ROPE", "1")
    monkeypatch.setattr(m, "_do_install", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    m.install()
    out = capsys.readouterr().out
    assert "install failed" in out and "active" not in out


def test_missing_rms_rope_leaves_no_half_install(monkeypatch):
    import comfy_kitchen.backends.eager as e
    before = _snapshot()
    monkeypatch.delattr(e, "rms_rope")
    with pytest.raises(AttributeError):
        m._do_install()
    assert not m._installed and not m._orig
    monkeypatch.undo()
    assert _snapshot() == before
