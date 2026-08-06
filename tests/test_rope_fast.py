import pytest, torch
from _patches import rope_fast_mps as m

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


@pytest.fixture(autouse=True)
def _install_rope_fast():
    m.install_for_test()                  # captures real eager originals into m._orig
    m._backend_events.clear()
    try:
        yield
    finally:
        m.uninstall_for_test()


def _orig_interleaved(x, fr):
    """PRIMARY oracle: the captured REAL eager apply_rope1 (computed before any poisoning)."""
    return m._orig["apply_rope1"](x, fr)


def _orig_split_half(x, fr):
    return m._orig["apply_rope_split_half1"](x, fr)


def _shapes():
    return [(1, 24, 4608, 128), (2, 16, 1024, 64), (1, 1, 37, 16)]   # all rank-4 [B,H,L,D]


# ---------------------------------------------------------------------------
# Task 2 — interleaved kernel vs REAL eager
# ---------------------------------------------------------------------------
@mps
@pytest.mark.parametrize("B,H,L,D", _shapes())
@pytest.mark.parametrize("dtype,atol,rtol", [
    (torch.float32, 2e-5, 2e-5),
    (torch.float16, 3e-3, 3e-3),
    (torch.bfloat16, 2e-2, 2e-2),
])
def test_interleaved_matches_real_eager(B, H, L, D, dtype, atol, rtol):
    torch.manual_seed(0)
    x = torch.randn(B, H, L, D, device="mps", dtype=dtype)
    fr = torch.randn(1, 1, L, D // 2, 2, 2, device="mps", dtype=torch.float32)
    ref = _orig_interleaved(x, fr)                      # PRIMARY oracle (real eager)
    m._backend_events.clear()
    out = m.apply_rope1_fused(x, fr)
    torch.mps.synchronize()
    assert m._last_backend() == "kernel", "fell back instead of running the kernel"
    assert out.dtype == dtype and out.shape == x.shape
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    # secondary diagnostic vs the formula helper:
    assert torch.allclose(out.float(), m._reference_interleaved(x, fr).float(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Task 3 — split-half kernel vs REAL eager
# ---------------------------------------------------------------------------
@mps
@pytest.mark.parametrize("B,H,L,D", _shapes())
@pytest.mark.parametrize("dtype,atol,rtol", [
    (torch.float32, 2e-5, 2e-5),
    (torch.bfloat16, 2e-2, 2e-2),
])
def test_split_half_matches_real_eager(B, H, L, D, dtype, atol, rtol):
    torch.manual_seed(1)
    x = torch.randn(B, H, L, D, device="mps", dtype=dtype)
    fr = torch.randn(1, 1, L, D // 2, 2, 2, device="mps", dtype=torch.float32)
    ref = _orig_split_half(x, fr)                       # PRIMARY oracle (real eager)
    m._backend_events.clear()
    out = m.apply_rope_split_half1_fused(x, fr)
    torch.mps.synchronize()
    assert m._last_backend() == "kernel"
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    assert torch.allclose(out.float(), m._reference_split_half(x, fr).float(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Task 4 — pair wrappers, cross-length, real public-API dispatch proof
# ---------------------------------------------------------------------------
@mps
def test_pair_cross_length():
    """xq (Lq) and xk (Lk) share one table; each slices its own prefix (real-eager parity).
    Per-call trace must show BOTH single-tensor calls took the kernel (MAJOR 5)."""
    torch.manual_seed(2)
    B, H, D = 1, 8, 64
    Lq, Lk = 4096, 512
    fr = torch.randn(1, 1, Lq, D // 2, 2, 2, device="mps", dtype=torch.float32)
    xq = torch.randn(B, H, Lq, D, device="mps", dtype=torch.bfloat16)
    xk = torch.randn(B, H, Lk, D, device="mps", dtype=torch.bfloat16)
    ref_q = _orig_interleaved(xq, fr)
    ref_k = _orig_interleaved(xk, fr)                   # real eager, pre-poison
    m._backend_events.clear()
    oq, ok = m.apply_rope_fused_pair(xq, xk, fr)
    torch.mps.synchronize()
    kinds = [ev[1] for ev in m._backend_events]
    assert kinds == ["kernel", "kernel"], f"both calls must be kernel, got {kinds}"
    assert torch.allclose(oq.float(), ref_q.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(ok.float(), ref_k.float(), atol=2e-2, rtol=2e-2)


@mps
@pytest.mark.parametrize("public_name", ["apply_rope1", "apply_rope",
                                         "apply_rope_split_half1", "apply_rope_split_half"])
def test_real_public_dispatch_hits_kernel(public_name):
    """BLOCKER 2: call the REAL comfy_kitchen public API (-> torch.ops -> custom op ->
    registry.get_implementation -> getattr(eager, name)) and prove our kernel fires. Poison the
    captured originals so any silent fallback RAISES. Assert kernel for all four ops."""
    import comfy_kitchen as ck
    torch.manual_seed(3)
    x = torch.randn(1, 4, 256, 64, device="mps", dtype=torch.bfloat16)
    fr = torch.randn(1, 1, 256, 32, 2, 2, device="mps", dtype=torch.float32)
    poison = lambda *a, **k: (_ for _ in ()).throw(AssertionError("fell back: kernel did not fire"))
    saved = dict(m._orig)
    for nm in m._orig:                          # poison ALL single-tensor + pair originals
        m._orig[nm] = poison
    m._backend_events.clear()
    try:
        if public_name == "apply_rope1":
            _ = ck.apply_rope1(x, fr)
        elif public_name == "apply_rope":
            _ = ck.apply_rope(x, x, fr)
        elif public_name == "apply_rope_split_half1":
            _ = ck.apply_rope_split_half1(x, fr)
        else:
            _ = ck.apply_rope_split_half(x, x, fr)
        torch.mps.synchronize()
    finally:
        m._orig.update(saved)
    kinds = [ev[1] for ev in m._backend_events]
    assert kinds and all(k == "kernel" for k in kinds), \
        f"{public_name}: expected kernel via real dispatch, got {kinds}"


# ---------------------------------------------------------------------------
# Task 5 — fallback regimes, dtype, cache spy
# ---------------------------------------------------------------------------
# (1) VALID-eager but UNSUPPORTED-kernel -> fallback SUCCEEDS and matches real eager.
@mps
def test_non_broadcast_table_falls_back_and_matches():
    """Non-broadcast leading table dims (real B/H) are valid eager but unsupported by the kernel
    (Open Q #2). Must fall back and produce the SAME result as real eager."""
    torch.manual_seed(7)
    B, H, L, D = 2, 4, 128, 64
    x = torch.randn(B, H, L, D, device="mps", dtype=torch.bfloat16)
    fr = torch.randn(B, H, L, D // 2, 2, 2, device="mps", dtype=torch.float32)   # leading dims != 1
    ref = _orig_interleaved(x, fr)               # real eager broadcasts B/H fine
    m._backend_events.clear()
    out = m.apply_rope1_fused(x, fr)
    assert m._last_backend() == "fallback"
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@mps
def test_non_fp32_table_falls_back_and_matches():
    """MAJOR 8: a bf16 freqs_cis table is valid eager but routed to fallback (kernel is fp32-only).
    Output must match real eager (which multiplies in the table dtype)."""
    torch.manual_seed(8)
    x = torch.randn(1, 8, 128, 64, device="mps", dtype=torch.bfloat16)
    fr = torch.randn(1, 1, 128, 32, 2, 2, device="mps", dtype=torch.bfloat16)
    ref = _orig_interleaved(x, fr)
    m._backend_events.clear()
    out = m.apply_rope1_fused(x, fr)
    assert m._last_backend() == "fallback"
    assert torch.allclose(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


# (2) GENUINELY MALFORMED -> kernel must NOT dispatch; real eager then legitimately RAISES.
@mps
def test_malformed_table_does_not_dispatch_and_raises():
    """BLOCKER 3 / fused-norm pattern: a wrong-shaped freqs_cis (halfD=16 vs D/2=32) cannot
    broadcast in real eager. The load-bearing assertion is that we did NOT dispatch the kernel
    (backend == 'fallback') and that the real eager fallback RAISES -- NOT an allclose."""
    torch.manual_seed(6)
    x = torch.randn(1, 8, 128, 64, device="mps", dtype=torch.bfloat16)   # D/2 = 32
    bad = torch.randn(1, 1, 128, 16, 2, 2, device="mps", dtype=torch.float32)  # halfD = 16 (malformed)
    m._backend_events.clear()
    m._backend_events.append(("stale", "kernel", ()))   # stale spy must be overwritten
    with pytest.raises(Exception):
        m.apply_rope1_fused(x, bad)                      # validation -> fallback -> real eager raises
    assert m._last_backend() == "fallback"


@mps
def test_per_call_trace_distinguishes_kernel_then_fallback():
    """MAJOR 5: the per-call trace must distinguish a kernel call from a fallback call -- a scalar
    `_last_backend` would only reflect the LAST and could mask an earlier mixed result.

    Note: a genuine [kernel, fallback] mix is NOT reachable through a single pair-wrapper call with
    a shared table, because the kernel's per-tensor rejections (rank!=4, odd-D, numel==0) are
    exactly the cases real eager also rejects, and the table-based rejections (non-fp32,
    non-broadcast) are shared by both tensors. So we drive the two paths with two sequential
    single-tensor calls: an fp32-table call (kernel) then a bf16-table call (valid eager, kernel
    fp32-only -> fallback). Both originals stay real so the fallback succeeds."""
    torch.manual_seed(9)
    x = torch.randn(1, 8, 256, 64, device="mps", dtype=torch.bfloat16)
    fr_fp32 = torch.randn(1, 1, 256, 32, 2, 2, device="mps", dtype=torch.float32)   # -> kernel
    fr_bf16 = torch.randn(1, 1, 256, 32, 2, 2, device="mps", dtype=torch.bfloat16)  # -> fallback
    m._backend_events.clear()
    _ = m.apply_rope1_fused(x, fr_fp32)
    _ = m.apply_rope1_fused(x, fr_bf16)
    kinds = [ev[1] for ev in m._backend_events]
    assert kinds == ["kernel", "fallback"], f"per-call trace must catch the split: {kinds}"
    # a scalar would only show the last ("fallback"); the list preserves both.
    assert m._last_backend() == "fallback"


@mps
def test_pair_wrapper_records_both_calls():
    """MAJOR 5 (pair form): a pair-wrapper call must append ONE event per tensor (two total), so a
    first-call fallback can never be masked by the second. Drive both to fallback with a bf16 table
    (shared) -> two 'fallback' events; combined with test_pair_cross_length's two 'kernel' events,
    this proves the trace is per-call, not a scalar."""
    torch.manual_seed(11)
    xq = torch.randn(1, 8, 256, 64, device="mps", dtype=torch.bfloat16)
    xk = torch.randn(1, 8, 128, 64, device="mps", dtype=torch.bfloat16)
    fr_bf16 = torch.randn(1, 1, 256, 32, 2, 2, device="mps", dtype=torch.bfloat16)
    ref_q = _orig_interleaved(xq, fr_bf16)
    ref_k = _orig_interleaved(xk, fr_bf16)
    m._backend_events.clear()
    oq, ok = m.apply_rope_fused_pair(xq, xk, fr_bf16)
    kinds = [ev[1] for ev in m._backend_events]
    assert kinds == ["fallback", "fallback"], f"pair must record one event per tensor: {kinds}"
    assert torch.allclose(oq.float(), ref_q.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(ok.float(), ref_k.float(), atol=2e-2, rtol=2e-2)


# (3) CPU input -> fallback (valid eager, no MPS dispatch).
def test_cpu_falls_back():
    x = torch.randn(1, 2, 8, 16)                   # CPU
    fr = torch.randn(1, 1, 8, 8, 2, 2)
    ref = m._orig["apply_rope1"](x, fr)
    m._backend_events.clear()
    out = m.apply_rope1_fused(x, fr)
    assert m._last_backend() == "fallback"
    assert torch.allclose(out.float(), ref.float(), atol=1e-5)


# (4) cache invalidation on in-place table mutation (MAJOR 9).
@mps
def test_table_mutation_invalidates_cache():
    """Mutating the table in place must change the output (no stale id()-keyed data)."""
    torch.manual_seed(10)
    x = torch.randn(1, 4, 128, 64, device="mps", dtype=torch.float32)
    fr = torch.randn(1, 1, 128, 32, 2, 2, device="mps", dtype=torch.float32)
    out1 = m.apply_rope1_fused(x, fr).clone()
    fr.mul_(2.0)                                         # in-place mutation bumps _version
    out2 = m.apply_rope1_fused(x, fr)
    ref2 = _orig_interleaved(x, fr)
    assert not torch.allclose(out1.float(), out2.float()), "cache returned stale table"
    assert torch.allclose(out2.float(), ref2.float(), atol=2e-5, rtol=2e-5)


@mps
def test_install_does_not_claim_active_when_the_reroute_failed(monkeypatch, capsys):
    """The banner is the only signal a user gets; it must not claim success
    when comfy_kitchen is missing or the eager reroute raised.

    install() has three early returns ahead of _do_install, so assert the stub
    actually ran. Without that this passes on any machine that trips a gate --
    proving nothing about the banner.
    """
    pytest.importorskip("comfy_kitchen.backends.eager")
    from _patches import _caps
    from _patches import rope_fast_mps as m

    called = []

    def boom():
        called.append(True)
        raise RuntimeError("no eager backend")

    monkeypatch.delenv("ASFP8_ROPE_FAST", raising=False)
    monkeypatch.setattr(_caps, "has_compile_shader", lambda: True)
    monkeypatch.setattr(m, "_do_install", boom)
    m.install()

    assert called, "install() returned before reaching _do_install; test proves nothing"
    out = capsys.readouterr().out
    assert "fused RoPE active" not in out, out
