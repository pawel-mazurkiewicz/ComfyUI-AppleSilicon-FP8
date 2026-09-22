"""Tests for _patches/int4_linear_mps.py (ConvRot W4A4 fast path on MPS).

Needs comfy-kitchen >= 0.2.13 for the ConvRot W4A4 layout, and skips otherwise.
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
    os.environ["ASFP8_INT4_EXT"] = "1"
    from _patches.int4_ext import loader
    # the default-off W4A16 tests above cache None here, so clear it or no build is
    # ever attempted
    loader._tried = False
    loader._mod = None
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


# --- per-kernel verification (issue #14) -----------------------------------


@requires_mps
def test_int4_self_check_passes_on_a_working_kernel(monkeypatch):
    """The self-check passes on a working kernel: warmup() only proves it compiles."""
    _kernel_or_skip()   # opts in and builds the extension, but not into _kernel
    monkeypatch.setattr(int4_linear_mps, "_kernel", None)
    monkeypatch.setattr(int4_linear_mps, "_kernel_tried", False)
    monkeypatch.setattr(int4_linear_mps, "_self_checked", False)
    monkeypatch.setattr(int4_linear_mps, "_self_ok", False)

    assert int4_linear_mps._load_kernel() is not None
    assert int4_linear_mps._self_check() is True


@requires_mps
def test_int4_self_check_rejects_a_wrong_kernel(monkeypatch):
    """Guards the guard: a self-check that cannot fail is worse than none."""
    _kernel_or_skip()

    class Wrong:
        def i8i4_linear_fused_nt(self, qx, w, xs, ws, b, K, N):
            return torch.zeros(qx.shape[0], N, dtype=torch.bfloat16, device="mps")

    monkeypatch.setattr(int4_linear_mps, "_kernel", Wrong())
    monkeypatch.setattr(int4_linear_mps, "_self_checked", False)
    monkeypatch.setattr(int4_linear_mps, "_self_ok", False)
    assert int4_linear_mps._self_check() is False


def test_int4_verification_is_memoised(monkeypatch):
    """A broken int4 kernel must not be re-verified on every ConvRot layer."""
    from _patches import _caps

    _caps._kernel_ready.pop("int4", None)
    attempts = []

    def failing_verify():
        attempts.append(1)
        return False

    try:
        for _ in range(3):
            assert _caps.kernel_ready("int4", failing_verify) is False
        assert len(attempts) == 1, f"int4 re-verified {len(attempts)}x"
    finally:
        _caps._kernel_ready.pop("int4", None)


@requires_mps
def test_int4_dispatch_failure_latches_off_the_kernel(monkeypatch):
    """A W4A8 dispatch failure latches the kernel off instead of retrying per layer (#13)."""
    from _patches import _caps

    calls = []

    def exploding(*a, **k):
        calls.append(1)
        raise RuntimeError("W4A8 dispatch blew up")

    _caps._kernel_ready.pop("int4", None)
    monkeypatch.setattr(int4_linear_mps, "_verify", lambda: True)
    monkeypatch.setattr(int4_linear_mps, "_load_kernel", lambda: object())
    monkeypatch.setattr(int4_linear_mps, "_w4a8_kernel_linear", exploding)

    _, qdata, wscales = _quantized_weight()
    x = torch.randn(8, K, dtype=torch.bfloat16, device="mps")

    try:
        for _ in range(4):
            out = int4_linear_mps._w4a16_linear_mps(
                x, qdata.to("mps"), wscales.to("mps"), None, 256, ck_eager
            )
            assert out.shape == (8, N)
        assert len(calls) == 1, f"W4A8 retried after a dispatch failure ({len(calls)}x)"
        assert _caps._kernel_ready["int4"] is False
    finally:
        _caps._kernel_ready.pop("int4", None)


@requires_mps
def test_int4_explicit_opt_in_is_honoured_on_pre_m5_hardware(monkeypatch):
    """An explicit ASFP8_INT4_EXT=1 is honoured on pre-M5 hardware (#25).

    int4 is opt-in, so a chip pre-filter would save no build and could only fire where
    the user asked for the kernel; the self-check still rejects a wrong result.
    """
    from _patches import _caps

    _caps._kernel_ready.pop("int4", None)
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: "Apple M4 Pro")
    monkeypatch.setenv("ASFP8_INT4_EXT", "1")

    verifies = []
    monkeypatch.setattr(int4_linear_mps, "_verify",
                        lambda: verifies.append(1) or False)

    _, qdata, wscales = _quantized_weight()
    x = torch.randn(8, K, dtype=torch.bfloat16, device="mps")

    try:
        out = int4_linear_mps._w4a16_linear_mps(
            x, qdata.to("mps"), wscales.to("mps"), None, 256, ck_eager
        )
        assert out.shape == (8, N), "W4A16 must still answer when the kernel is out"
        assert verifies == [1], (
            "an explicit ASFP8_INT4_EXT=1 was overridden by the chip probe"
        )
    finally:
        _caps._kernel_ready.pop("int4", None)
