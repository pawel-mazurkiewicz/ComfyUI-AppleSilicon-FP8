"""Tests for _patches/int4_linear_mps.py (ConvRot W4A4 fast path on MPS).

Requires comfy-kitchen >= 0.2.13 (ConvRot W4A4 layout). Skips otherwise —
run with PYTHONPATH pointing at a new-enough kitchen if the env has an old one.
"""

import pytest
import torch

from conftest import requires_mps

ck_eager = pytest.importorskip("comfy_kitchen.backends.eager.convrot_w4a4")

from _patches import int4_linear_mps  # noqa: E402

M, K, N = 128, 1024, 512  # K divisible by 256 (convrot) and 64 (quant group)


def _quantized_weight(seed=0):
    torch.manual_seed(seed)
    w = torch.randn(N, K, dtype=torch.float32) * 0.02
    qdata, wscales = ck_eager.quantize_convrot_w4a4_weight(w, convrot_groupsize=256)
    return w, qdata, wscales


def test_unpack_bit_exact_cpu():
    p = torch.randint(-128, 128, (64, 128), dtype=torch.int8)
    ref = ck_eager._unpack_int4_row_major(p).to(torch.int8)
    got = int4_linear_mps._unpack_int4_signed_fast(p, torch.int8)
    assert torch.equal(ref, got)


@requires_mps
def test_unpack_bit_exact_mps():
    p = torch.randint(-128, 128, (64, 128), dtype=torch.int8, device="mps")
    ref = ck_eager._unpack_int4_row_major(p.cpu()).to(torch.int8)
    got = int4_linear_mps._unpack_int4_signed_fast(p, torch.int8).cpu()
    assert torch.equal(ref, got)


@requires_mps
def test_w4a16_more_accurate_than_eager():
    w, qdata, wscales = _quantized_weight()
    torch.manual_seed(1)
    x = torch.randn(M, K, dtype=torch.float32)
    ref = torch.nn.functional.linear(x, w)

    y_eager = ck_eager.convrot_w4a4_linear(x.to("mps", torch.bfloat16), qdata.to("mps"),
                                           wscales.to("mps"), convrot_groupsize=256)
    y_fast = int4_linear_mps._w4a16_linear_mps(x.to("mps", torch.bfloat16), qdata.to("mps"),
                                               wscales.to("mps"), None, 256, ck_eager)

    err_eager = (y_eager.float().cpu() - ref).abs().mean() / ref.abs().mean()
    err_fast = (y_fast.float().cpu() - ref).abs().mean() / ref.abs().mean()
    assert err_fast < err_eager, f"fast path err {err_fast} not < eager err {err_eager}"
    assert err_fast < 0.25  # int4 weight-only on random gaussian data


@requires_mps
def test_w4a16_bias_and_3d_input():
    w, qdata, wscales = _quantized_weight()
    bias = torch.randn(N, dtype=torch.float32)
    x = torch.randn(2, 64, K, dtype=torch.bfloat16, device="mps")
    y = int4_linear_mps._w4a16_linear_mps(x, qdata.to("mps"), wscales.to("mps"),
                                          bias.to("mps"), 256, ck_eager)
    assert y.shape == (2, 64, N)
    ref = torch.nn.functional.linear(
        x.float().cpu().reshape(-1, K), w, bias).reshape(2, 64, N)
    err = (y.float().cpu() - ref).abs().mean() / ref.abs().mean()
    assert err < 0.25


def _kernel_or_skip():
    import os
    os.environ.setdefault("ASFP8_INT4_EXT", "1")
    from _patches.int4_ext import loader
    mod = loader.module()
    if mod is None:
        pytest.skip("int4 Metal extension unavailable (build failed or no toolchain)")
    return mod


@requires_mps
def test_w4a8_kernel_bit_exact_vs_emulation():
    kernel = _kernel_or_skip()
    torch.manual_seed(2)
    Mk = 200  # non-multiple of tile to exercise bounds
    _, qdata, wscales = _quantized_weight()
    x_rot = torch.randn(Mk, K, dtype=torch.bfloat16, device="mps")
    bias = torch.randn(N, dtype=torch.bfloat16, device="mps")

    from _patches import int4_linear_mps
    y = int4_linear_mps._w4a8_kernel_linear(x_rot, qdata.to("mps"), wscales.to("mps"),
                                            bias, kernel)

    # emulate: same act quant, int32 matmul on unpacked weight, same epilogue
    absmax = x_rot.float().abs().amax(dim=-1).clamp(min=1e-10)
    x_scale = absmax / 127.0
    qx = torch.round(x_rot.float() / x_scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    w_int = ck_eager._unpack_int4_row_major(qdata).to(torch.int32)
    acc = qx.cpu().to(torch.int32) @ w_int.T
    ref = (acc.float() * x_scale.cpu().unsqueeze(-1) * wscales.reshape(1, -1)).to(torch.bfloat16)
    ref = ref + bias.cpu()
    assert torch.equal(y.cpu(), ref), \
        f"kernel mismatch: maxdiff={(y.cpu().float() - ref.float()).abs().max()}"


@requires_mps
def test_w4a8_kernel_no_bias():
    kernel = _kernel_or_skip()
    torch.manual_seed(3)
    _, qdata, wscales = _quantized_weight()
    x_rot = torch.randn(64, K, dtype=torch.bfloat16, device="mps")
    from _patches import int4_linear_mps
    y = int4_linear_mps._w4a8_kernel_linear(x_rot, qdata.to("mps"), wscales.to("mps"),
                                            None, kernel)
    assert y.shape == (64, N)
    assert torch.isfinite(y.float()).all()


@requires_mps
def test_install_routes_mps_only():
    import comfy_kitchen.tensor.convrot_w4a4 as ck_tensor

    int4_linear_mps.install()
    try:
        assert int4_linear_mps._patched
        w, qdata, wscales = _quantized_weight()
        x = torch.randn(M, K, dtype=torch.bfloat16)

        # CPU falls through to the original eager path (identical result)
        y_cpu = ck_tensor.convrot_w4a4_linear(x, qdata, wscales, convrot_groupsize=256)
        y_orig = int4_linear_mps._orig(x, qdata, wscales, convrot_groupsize=256)
        assert torch.equal(y_cpu, y_orig)

        # MPS goes through the fast path (matches direct call)
        y_mps = ck_tensor.convrot_w4a4_linear(x.to("mps"), qdata.to("mps"),
                                              wscales.to("mps"), convrot_groupsize=256)
        y_fast = int4_linear_mps._w4a16_linear_mps(x.to("mps"), qdata.to("mps"),
                                                   wscales.to("mps"), None, 256, ck_eager)
        assert torch.equal(y_mps.cpu(), y_fast.cpu())
    finally:
        int4_linear_mps.uninstall()
