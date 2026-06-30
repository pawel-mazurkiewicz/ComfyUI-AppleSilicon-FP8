"""Patch #21b spy tests: the fused RoPE kernel retargeted onto the REAL comfy functions the live
models call (probe 2026-07-01):

  comfy.ldm.flux.math.apply_rope / apply_rope1      -- interleaved 2x2 (split=0)
  comfy.text_encoders.llama.apply_rope              -- half-split rotate_half, (cos,sin,nsin) tuple
  comfy.ldm.ideogram4.model.apply_rope              -- SAME object as llama's (alias import)

Oracle = the captured REAL comfy function (m._orig_comfy[...]), computed before the kernel runs.
Each kernel test also POISONS the captured original so a silent fallback would RAISE -- proving the
fused kernel actually fired. Requires comfy on sys.path; otherwise the whole module is skipped."""
import os
import sys
import pytest
import torch

# Locate the installed comfy package (machine-specific; overridable via env). ComfyUI-desktop keeps
# the code tree outside the repo venv, so it is not importable by default.
_CANDIDATES = [
    os.environ.get("ASFP8_COMFY_PATH"),
    "/Users/pawelma/ComfyUI-Installs/ComfyUI/ComfyUI",
]
for _c in _CANDIDATES:
    if _c and os.path.isdir(os.path.join(_c, "comfy")) and _c not in sys.path:
        sys.path.insert(0, _c)

fm = pytest.importorskip("comfy.ldm.flux.math")
ll = pytest.importorskip("comfy.text_encoders.llama")
from comfy.text_encoders.llama import precompute_freqs_cis  # noqa: E402

from _patches import rope_fast_mps as m  # noqa: E402

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


@pytest.fixture
def comfy_patched():
    """Capture + patch the comfy targets, then restore. Leaves m._orig_comfy populated with the REAL
    functions (the oracle) for the duration of the test."""
    m._backend_events.clear()
    m._install_comfy()
    try:
        yield
    finally:
        m._uninstall_comfy()
        m._backend_events.clear()


def _flux_pe(L, D, device, theta=10000):
    """comfy flux rotation table: rope()->(1,L,halfD,2,2); unsqueeze(1) for head-broadcast as the
    live probe saw (1,1,L,halfD,2,2)."""
    pos = torch.arange(L, device=device).float()[None, :]
    return fm.rope(pos, D, theta).unsqueeze(1)


def _events(name):
    return [e for e in m._backend_events if e[0] == name]


# ---------------------------------------------------------------------------
# FLUX -- comfy.ldm.flux.math.apply_rope (interleaved 2x2, split=0)
# ---------------------------------------------------------------------------
@mps
@pytest.mark.parametrize("dtype,atol,rtol", [
    (torch.float32, 2e-5, 2e-5),
    (torch.bfloat16, 2e-2, 2e-2),
])
def test_flux_apply_rope_matches_real(comfy_patched, dtype, atol, rtol):
    torch.manual_seed(0)
    B, H, L, D = 1, 32, 512, 128          # probe shape, shorter L for test speed
    x = torch.randn(B, H, L, D, device="mps", dtype=dtype)
    pe = _flux_pe(L, D, "mps")
    ref_q, ref_k = m._orig_comfy["flux.apply_rope"](x.clone(), x.clone(), pe)   # REAL oracle
    m._backend_events.clear()
    out_q, out_k = fm.apply_rope(x.clone(), x.clone(), pe)                      # patched -> kernel
    torch.mps.synchronize()
    kinds = [e[1] for e in _events("flux.apply_rope")]
    assert kinds == ["kernel", "kernel"], f"flux pair must run kernel for both, got {kinds}"
    assert out_q.dtype == dtype and out_q.shape == x.shape
    assert torch.allclose(out_q.float(), ref_q.float(), atol=atol, rtol=rtol)
    assert torch.allclose(out_k.float(), ref_k.float(), atol=atol, rtol=rtol)


@mps
def test_flux_apply_rope1_matches_real(comfy_patched):
    torch.manual_seed(1)
    B, H, L, D = 1, 24, 256, 128
    x = torch.randn(B, H, L, D, device="mps", dtype=torch.bfloat16)
    pe = _flux_pe(L, D, "mps")
    ref = m._orig_comfy["flux.apply_rope1"](x.clone(), pe)
    m._backend_events.clear()
    out = fm.apply_rope1(x.clone(), pe)
    torch.mps.synchronize()
    assert [e[1] for e in _events("flux.apply_rope1")] == ["kernel"]
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


@mps
def test_flux_kernel_actually_fires_poison(comfy_patched):
    """Poison the captured REAL flux original so any fallback RAISES; the call must still succeed
    (kernel fired) and the trace must show kernel."""
    torch.manual_seed(2)
    x = torch.randn(1, 32, 256, 128, device="mps", dtype=torch.bfloat16)
    pe = _flux_pe(256, 128, "mps")
    poison = lambda *a, **k: (_ for _ in ()).throw(AssertionError("flux fell back: kernel did not fire"))
    saved = dict(m._orig_comfy)
    m._orig_comfy["flux.apply_rope"] = poison
    m._orig_comfy["flux.apply_rope1"] = poison
    m._backend_events.clear()
    try:
        out_q, out_k = fm.apply_rope(x.clone(), x.clone(), pe)
        torch.mps.synchronize()
    finally:
        m._orig_comfy.update(saved)
    assert [e[1] for e in _events("flux.apply_rope")] == ["kernel", "kernel"]


# ---------------------------------------------------------------------------
# LLAMA / IDEOGRAM4 -- comfy.text_encoders.llama.apply_rope (half-split, (cos,sin,nsin))
# ---------------------------------------------------------------------------
@mps
@pytest.mark.parametrize("dtype,atol,rtol", [
    (torch.float32, 2e-5, 2e-5),
    (torch.float16, 3e-3, 3e-3),
    (torch.bfloat16, 2e-2, 2e-2),
])
@pytest.mark.parametrize("head_dim,seq,heads", [(128, 256, 8), (64, 512, 16)])
def test_llama_apply_rope_matches_real(comfy_patched, dtype, atol, rtol, head_dim, seq, heads):
    torch.manual_seed(3)
    pos_ids = torch.arange(seq, device="mps")[None, :]
    freqs = precompute_freqs_cis(head_dim, pos_ids, 10000.0, device="mps")
    assert isinstance(freqs, tuple) and len(freqs) == 3      # (cos, sin, nsin)
    xq = torch.randn(1, heads, seq, head_dim, device="mps", dtype=dtype)
    xk = torch.randn(1, heads, seq, head_dim, device="mps", dtype=dtype)
    ref_q, ref_k = m._orig_comfy["llama.apply_rope"](xq.clone(), xk.clone(), freqs)   # REAL oracle
    m._backend_events.clear()
    out_q, out_k = ll.apply_rope(xq.clone(), xk.clone(), freqs)
    torch.mps.synchronize()
    kinds = [e[1] for e in _events("llama.apply_rope")]
    assert kinds == ["kernel", "kernel"], f"llama pair must run kernel for both, got {kinds}"
    assert out_q.dtype == dtype and out_q.shape == xq.shape
    assert torch.allclose(out_q.float(), ref_q.float(), atol=atol, rtol=rtol)
    assert torch.allclose(out_k.float(), ref_k.float(), atol=atol, rtol=rtol)


@mps
def test_llama_kernel_actually_fires_poison(comfy_patched):
    torch.manual_seed(4)
    head_dim, seq, heads = 128, 256, 8
    pos_ids = torch.arange(seq, device="mps")[None, :]
    freqs = precompute_freqs_cis(head_dim, pos_ids, 10000.0, device="mps")
    xq = torch.randn(1, heads, seq, head_dim, device="mps", dtype=torch.bfloat16)
    xk = torch.randn(1, heads, seq, head_dim, device="mps", dtype=torch.bfloat16)
    poison = lambda *a, **k: (_ for _ in ()).throw(AssertionError("llama fell back: kernel did not fire"))
    saved = dict(m._orig_comfy)
    m._orig_comfy["llama.apply_rope"] = poison
    m._backend_events.clear()
    try:
        _ = ll.apply_rope(xq, xk, freqs)
        torch.mps.synchronize()
    finally:
        m._orig_comfy.update(saved)
    assert [e[1] for e in _events("llama.apply_rope")] == ["kernel", "kernel"]


@mps
def test_ideogram4_alias_is_retargeted():
    """comfy.ldm.ideogram4.model.apply_rope IS llama.apply_rope (alias import). _install_comfy must
    reroute it by object identity, and the rerouted fn must run the kernel and match the real llama
    apply_rope."""
    ig = pytest.importorskip("comfy.ldm.ideogram4.model")
    assert ig.apply_rope is ll.apply_rope, "precondition: ideogram4 imports llama's apply_rope"
    real = ll.apply_rope
    m._backend_events.clear()
    m._install_comfy()
    try:
        assert ig.apply_rope is m._llama_apply_rope_fused, "alias not rerouted by identity"
        assert ig.apply_rope is not real
        head_dim, seq, heads = 128, 128, 4
        pos_ids = torch.arange(seq, device="mps")[None, :]
        freqs = precompute_freqs_cis(head_dim, pos_ids, 10000.0, device="mps")
        xq = torch.randn(1, heads, seq, head_dim, device="mps", dtype=torch.bfloat16)
        xk = torch.randn(1, heads, seq, head_dim, device="mps", dtype=torch.bfloat16)
        ref_q, ref_k = m._orig_comfy["llama.apply_rope"](xq.clone(), xk.clone(), freqs)
        m._backend_events.clear()
        out_q, out_k = ig.apply_rope(xq.clone(), xk.clone(), freqs)   # via ideogram alias
        torch.mps.synchronize()
        assert [e[1] for e in _events("llama.apply_rope")] == ["kernel", "kernel"]
        assert torch.allclose(out_q.float(), ref_q.float(), atol=2e-2, rtol=2e-2)
        assert torch.allclose(out_k.float(), ref_k.float(), atol=2e-2, rtol=2e-2)
    finally:
        m._uninstall_comfy()
        # identity restore
        assert ig.apply_rope is real, "uninstall must restore the alias"


# ---------------------------------------------------------------------------
# Fallback regimes -- valid comfy, unsupported kernel -> fall back & match real
# ---------------------------------------------------------------------------
@mps
def test_llama_true_batch_falls_back_and_matches(comfy_patched):
    """A real batch>1 freqs table (cos rows = B*L != L) cannot use the kernel's row%L indexing.
    Must fall back to the real comfy apply_rope and match it exactly."""
    torch.manual_seed(5)
    head_dim, seq, heads, B = 128, 64, 4, 2
    pos_ids = torch.arange(seq, device="mps")[None, :].expand(B, -1).contiguous()
    freqs = precompute_freqs_cis(head_dim, pos_ids, 10000.0, device="mps")
    xq = torch.randn(B, heads, seq, head_dim, device="mps", dtype=torch.bfloat16)
    xk = torch.randn(B, heads, seq, head_dim, device="mps", dtype=torch.bfloat16)
    ref_q, ref_k = m._orig_comfy["llama.apply_rope"](xq.clone(), xk.clone(), freqs)
    m._backend_events.clear()
    out_q, out_k = ll.apply_rope(xq.clone(), xk.clone(), freqs)
    assert [e[1] for e in _events("llama.apply_rope")] == ["fallback"]
    assert torch.allclose(out_q.float(), ref_q.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(out_k.float(), ref_k.float(), atol=2e-2, rtol=2e-2)


@mps
def test_flux_non_broadcast_table_falls_back_and_matches(comfy_patched):
    """A flux pe with real (non-1) batch/head leading dims is valid comfy but unsupported by the
    kernel -> fall back and match."""
    torch.manual_seed(6)
    B, H, L, D = 2, 4, 128, 64
    x = torch.randn(B, H, L, D, device="mps", dtype=torch.bfloat16)
    pe = torch.randn(B, H, L, D // 2, 2, 2, device="mps", dtype=torch.float32)  # leading dims != 1
    ref_q, ref_k = m._orig_comfy["flux.apply_rope"](x.clone(), x.clone(), pe)
    m._backend_events.clear()
    out_q, out_k = fm.apply_rope(x.clone(), x.clone(), pe)
    assert [e[1] for e in _events("flux.apply_rope")] == ["fallback"]
    assert torch.allclose(out_q.float(), ref_q.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(out_k.float(), ref_k.float(), atol=2e-2, rtol=2e-2)
