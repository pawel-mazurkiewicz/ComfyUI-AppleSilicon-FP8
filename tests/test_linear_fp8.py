"""Tests for patch #10: F.linear FP8 operand decode on MPS."""

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_mps

from _patches import linear_fp8
from _patches._common import decode_fp8


def _cpu_ref(x, w, b, out_dtype):
    """CPU float32 reference for F.linear."""
    out = F.linear(x.float(), w.float(), b.float() if b is not None else None)
    return out.to(out_dtype)


@requires_mps
def test_fp8_weight_bf16_input_no_bias():
    """Most common case: fp8 weight, bf16 input, no bias."""
    torch.manual_seed(0)
    M, K, N = 8, 32, 16
    x = (torch.randn(M, K) * 0.3).to(torch.bfloat16).to("mps")
    w = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn)
    ref = _cpu_ref(x.cpu(), w, None, torch.bfloat16)

    out = linear_fp8._patched_linear(x, w.to("mps"), None)

    assert out.dtype == torch.bfloat16
    assert out.device.type == "mps"
    rel = (out.cpu().float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-9)
    assert rel < 0.05, f"rel error {rel:.4f}"


@requires_mps
def test_fp8_weight_bf16_input_with_bias():
    """fp8 weight + bf16 bias + bf16 input."""
    torch.manual_seed(1)
    M, K, N = 8, 32, 16
    x = (torch.randn(M, K) * 0.3).to(torch.bfloat16).to("mps")
    w = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn).to("mps")
    b = (torch.randn(N) * 0.1).to(torch.bfloat16).to("mps")
    ref = _cpu_ref(x.cpu(), w.cpu(), b.cpu(), torch.bfloat16)

    out = linear_fp8._patched_linear(x, w, b)

    assert out.dtype == torch.bfloat16
    rel = (out.cpu().float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-9)
    assert rel < 0.05, f"rel error {rel:.4f}"


@requires_mps
def test_fp8_input_fp8_weight():
    """Both input and weight are fp8 — decode both to bf16."""
    torch.manual_seed(2)
    M, K, N = 8, 32, 16
    x = (torch.randn(M, K) * 0.3).to(torch.float8_e4m3fn).to("mps")
    w = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn).to("mps")
    # Reference: decode both to float32 on CPU
    ref = F.linear(decode_fp8(x.cpu(), torch.float32), decode_fp8(w.cpu(), torch.float32))

    out = linear_fp8._patched_linear(x, w, None)

    assert out.dtype == torch.bfloat16
    rel = (out.cpu().float() - ref).abs().max() / (ref.abs().max() + 1e-9)
    assert rel < 0.05, f"rel error {rel:.4f}"


@requires_mps
def test_non_fp8_fast_path():
    """Non-FP8 inputs take the fast path and return the same result as the original."""
    torch.manual_seed(3)
    x = torch.randn(8, 32, device="mps", dtype=torch.bfloat16)
    w = torch.randn(16, 32, device="mps", dtype=torch.bfloat16)
    b = torch.randn(16, device="mps", dtype=torch.bfloat16)

    expected = F.linear(x, w, b)
    out = linear_fp8._patched_linear(x, w, b)

    assert torch.equal(out, expected), "fast path should be bit-identical to original"


@requires_mps
def test_fp8_weight_f32_input():
    """fp8 weight, f32 input — decode weight to f32 to match input dtype."""
    torch.manual_seed(4)
    M, K, N = 8, 32, 16
    x = (torch.randn(M, K) * 0.3).to(torch.float32).to("mps")
    w = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn).to("mps")
    ref = _cpu_ref(x.cpu(), w.cpu(), None, torch.float32)

    out = linear_fp8._patched_linear(x, w, None)

    assert out.dtype == torch.float32
    rel = (out.cpu() - ref).abs().max() / (ref.abs().max() + 1e-9)
    assert rel < 0.05, f"rel error {rel:.4f}"


@requires_mps
def test_install_patches_f_linear():
    """After install(), F.linear handles fp8 weight on MPS without raising."""
    linear_fp8.install()

    torch.manual_seed(5)
    x = (torch.randn(4, 16) * 0.3).to(torch.bfloat16).to("mps")
    w = (torch.randn(8, 16) * 0.3).to(torch.float8_e4m3fn).to("mps")

    # This would raise RuntimeError on unpatched MPS
    out = F.linear(x, w)
    assert out.shape == (4, 8)
    assert out.dtype == torch.bfloat16
