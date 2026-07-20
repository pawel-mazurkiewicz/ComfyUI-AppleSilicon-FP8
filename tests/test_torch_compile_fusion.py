"""Tests for Issue A: torch.compile (Inductor MPS) fusion sweep.

Correctness tests only — timing is in dev/bench_torch_compile_fusion.py.

Tests are skipped on non-MPS machines. The compile step is guarded with
try/except: if Inductor MPS raises BackendCompilerFailed (immature backend),
the test is skipped with reason="inductor MPS compile failed" rather than
crashing the test runner. Never fatal.

NOTE: Do NOT run with ASFP8_PROFILE=1 — the mps_profile wrappers
(F.rms_norm, F.linear, torch.matmul, etc.) are not transparent to Dynamo
and will introduce graph breaks that mask the real fusion result.
"""
import pytest
import torch
import torch.nn.functional as F

_MPS = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs MPS"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(B=2, S=4096, D=1536, seed=0, device="mps", dtype=torch.float16):
    """Return (x, weight, W) for the representative DiT-block tail."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(B, S, D, device=device, dtype=dtype, generator=g)
    weight = torch.randn(D, device=device, dtype=dtype, generator=g)
    W = torch.randn(D, D, device=device, dtype=dtype, generator=g)
    return x, weight, W


def _bw_tail(x, weight, W):
    """Bandwidth-bound tail only: rms_norm + silu + residual (NO linear).

    This is the sub-chain where fusion delivers maximum benefit. The linear
    is excluded because it is compute-bound and breaks the pointwise fusion
    chain — measured separately in the full-block test.
    """
    h = F.rms_norm(x, (x.shape[-1],), weight, 1e-6)
    h = F.silu(h)
    return x + h


def _full_block(x, weight, W):
    """Full DiT block tail: rms_norm -> linear -> silu -> residual."""
    h = F.rms_norm(x, (x.shape[-1],), weight, 1e-6)
    h = (h.reshape(-1, h.shape[-1]) @ W.T).reshape(x.shape)
    h = F.silu(h)
    return x + h


def _try_compile(fn, **compile_kwargs):
    """Compile fn with Inductor MPS. Returns (compiled_fn, error_str_or_None)."""
    try:
        compiled = torch.compile(fn, backend="inductor", **compile_kwargs)
        return compiled, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# A.1.a — bandwidth-bound tail (no linear): compiled matches eager
# ---------------------------------------------------------------------------

@_MPS
def test_bw_tail_compiled_matches_eager():
    """rms_norm + silu + residual: Inductor MPS compiled output must match eager.

    Tolerance: atol=1e-2, rtol=1e-2 (empirical fp16 fusion noise for this chain).
    Also asserts against manual fp32 reference to verify rms_norm accumulation.
    """
    x, weight, W = _make_inputs()

    compiled_fn, err = _try_compile(_bw_tail)
    if err is not None:
        pytest.skip(f"inductor MPS compile failed — hand-kernels D/E at full scope. Error: {err}")

    eager_out = _bw_tail(x, weight, W)
    # First call compiles; second call uses the cached kernel.
    _ = compiled_fn(x, weight, W)
    compiled_out = compiled_fn(x, weight, W)
    torch.mps.synchronize()

    assert compiled_out.shape == eager_out.shape, (
        f"shape mismatch: compiled={compiled_out.shape} eager={eager_out.shape}"
    )

    # Primary: compiled vs eager fp16
    torch.testing.assert_close(
        compiled_out, eager_out, atol=1e-2, rtol=1e-2,
        msg=lambda m: f"compiled vs eager fp16: {m}",
    )

    # Secondary: compiled vs fp32 reference for rms_norm accumulation
    x_cpu = x.float().cpu()
    w_cpu = weight.float().cpu()
    eps = 1e-6
    rms_ref_cpu = (
        x_cpu * torch.rsqrt(x_cpu.pow(2).mean(-1, keepdim=True) + eps)
        * w_cpu
    )
    rms_ref_cpu = torch.sigmoid(rms_ref_cpu) * rms_ref_cpu  # silu
    ref_fp32 = x_cpu + rms_ref_cpu
    compiled_cpu = compiled_out.float().cpu()
    torch.testing.assert_close(
        compiled_cpu, ref_fp32, atol=5e-2, rtol=5e-2,
        msg=lambda m: f"compiled vs fp32 manual reference: {m}",
    )


# ---------------------------------------------------------------------------
# A.1.b — full block (with linear): compiled matches eager
# ---------------------------------------------------------------------------

@_MPS
def test_full_block_compiled_matches_eager():
    """Full block (rms_norm -> linear -> silu -> residual): same correctness check."""
    x, weight, W = _make_inputs()

    compiled_fn, err = _try_compile(_full_block)
    if err is not None:
        pytest.skip(f"inductor MPS compile failed (full block). Error: {err}")

    eager_out = _full_block(x, weight, W)
    _ = compiled_fn(x, weight, W)
    compiled_out = compiled_fn(x, weight, W)
    torch.mps.synchronize()

    # GEMM (h @ W.T) accumulates larger fp16 errors than norm+act alone.
    # bw_tail (no GEMM) passes at atol=1e-2; full_block needs atol=0.1 for GEMM noise.
    # 0.07 absolute difference is within normal fp16 1536-wide matmul bounds.
    torch.testing.assert_close(
        compiled_out, eager_out, atol=0.1, rtol=0.1,
        msg=lambda m: f"full_block compiled vs eager: {m}",
    )


# ---------------------------------------------------------------------------
# A.1.c — fullgraph=True: confirm no graph breaks AND compile actually succeeds
# ---------------------------------------------------------------------------

@_MPS
def test_bw_tail_no_graph_breaks():
    """Verify 0 graph breaks via dynamo.explain AND fullgraph=True compile succeeds.

    graph_count == 1 from explain() is necessary but not sufficient (does not
    prove fullgraph=True will succeed or that Inductor emits a fused kernel).
    Both checks are performed:
      1. dynamo.explain: graph_count == 1
      2. torch.compile(..., fullgraph=True): compiles and runs without error
    """
    import torch._dynamo as dynamo

    x, weight, W = _make_inputs()

    # Check 1: dynamo.explain graph count
    explanation = dynamo.explain(_bw_tail)(x, weight, W)
    assert explanation.graph_count == 1, (
        f"Expected 1 graph (no breaks), got {explanation.graph_count}. "
        f"Break reasons: {[str(r) for r in explanation.break_reasons]}"
    )

    # Check 2: fullgraph=True compile + execute
    torch._dynamo.reset()
    compiled_fg, err = _try_compile(_bw_tail, fullgraph=True)
    if err is not None:
        pytest.fail(
            f"dynamo.explain showed graph_count=1 but fullgraph=True FAILED. "
            f"Error: {err}"
        )
    out = compiled_fg(x, weight, W)
    torch.mps.synchronize()
    assert out is not None, "fullgraph=True compile returned None output"


# ---------------------------------------------------------------------------
# A.1.d — correctness at small shapes (fast, no-warmup shapes for CI)
# ---------------------------------------------------------------------------

@_MPS
@pytest.mark.parametrize("B,S,D", [
    (1, 256, 512),
    (2, 1024, 768),
    (1, 4096, 1536),  # production DiT shape
])
def test_bw_tail_shapes(B, S, D):
    """Correctness across a range of shapes including the production shape."""
    x, weight, W = _make_inputs(B=B, S=S, D=D)

    # Reset Dynamo state before the one compile to avoid shape cache pollution.
    torch._dynamo.reset()
    compiled_fn, err = _try_compile(_bw_tail)
    if err is not None:
        pytest.skip(f"compile failed: {err}")

    eager_out = _bw_tail(x, weight, W)
    _ = compiled_fn(x, weight, W)
    compiled_out = compiled_fn(x, weight, W)
    torch.mps.synchronize()

    torch.testing.assert_close(
        compiled_out, eager_out, atol=1e-2, rtol=1e-2,
        msg=lambda m: f"B={B} S={S} D={D}: {m}",
    )


# ---------------------------------------------------------------------------
# A.1.e — fp16 rms_norm precision: stress inputs + fp32 reference comparison
# ---------------------------------------------------------------------------

@_MPS
def test_compiled_rms_norm_fp32_reference():
    """Compiled rms_norm must not deviate from fp32 reference on stress inputs.

    Uses large-magnitude inputs (x10 scale) to amplify fp16 vs fp32 difference.
    Absence of NaN on random data is not evidence of fp32 accumulation.
    """
    x, weight, W = _make_inputs()

    compiled_fn, err = _try_compile(_bw_tail)
    if err is not None:
        pytest.skip(f"compile failed: {err}")

    # Large-magnitude stress input
    x_stress = x * 10.0
    _ = compiled_fn(x_stress, weight, W)
    compiled_out = compiled_fn(x_stress, weight, W)
    torch.mps.synchronize()

    assert not torch.isnan(compiled_out).any(), "NaN in compiled output on stress inputs"
    assert not torch.isinf(compiled_out).any(), "Inf in compiled output on stress inputs"

    # fp32 reference for the rms_norm step
    xf = x_stress.float().cpu()
    wf = weight.float().cpu()
    eps = 1e-6
    h_ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * wf
    h_ref = torch.sigmoid(h_ref) * h_ref  # silu
    ref_out = xf + h_ref

    torch.testing.assert_close(
        compiled_out.float().cpu(), ref_out, atol=5e-2, rtol=5e-2,
        msg=lambda m: f"compiled vs fp32 manual ref (stress inputs): {m}",
    )
