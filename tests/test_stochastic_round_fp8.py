"""Tests for patch #7: FP8 re-quant on MPS (LoRA applied to an fp8 base model).

Issue #29: loading an fp8 DiT with a LoRA took 80-200s, while the same model with
no LoRA loaded fast and int8 convrot was unaffected. LoRA application re-quantises
every weight it touches, and this patch was sending each of those to the CPU in
full -- because ONE operation inside the re-quant, the final float->fp8 cast, has
no MPS kernel. Patch #8 (tensor_to_fp8) already routes exactly that cast via a
LUT/CPU hop, so the rest of the maths can stay on the GPU. Measured 5.6x at
per-weight granularity (11.3 -> 2.0 ms/weight), bit-exact.

The CPU round-trip has to stay reachable: comfy's own fallback implementation
writes through `output[i:].copy_(...)`, a strided fp8 copy that patch #8 does not
cover (it wraps `.to`, not `copy_`).
"""
import torch

import pytest

from conftest import requires_mps

from _patches import stochastic_round_fp8 as patch


@pytest.fixture(autouse=True)
def _forget_native_probe(monkeypatch):
    """The native/CPU verdict is memoised per session; don't leak it between tests."""
    monkeypatch.setattr(patch, "_native_ok", None, raising=False)


def _recorder(fail_on_mps=False):
    """Stand-in for comfy.float.stochastic_rounding that records operand devices."""
    seen = []

    def original(value, dtype, seed=0):
        seen.append(value.device.type)
        if fail_on_mps and value.device.type == "mps":
            raise TypeError("Trying to convert Float8_e4m3fn to the MPS backend")
        return torch.zeros(value.shape, dtype=dtype, device=value.device)

    original.seen = seen
    return original


@requires_mps
def test_native_mps_path_is_preferred():
    """The whole point of #29: don't ship the tensor to the CPU when the GPU can
    do the maths and patch #8 can handle the one unsupported cast."""
    original = _recorder()
    x = torch.randn(256, device="mps")

    out = patch._requant(original, x, torch.float8_e4m3fn, 0)

    assert original.seen == ["mps"], f"re-quant left the GPU: {original.seen}"
    assert out.device.type == "mps"


@requires_mps
def test_falls_back_to_the_cpu_round_trip_when_native_raises():
    """comfy's own fallback implementation writes through a strided fp8 copy_,
    which patch #8 doesn't cover -- that path still needs the CPU hop."""
    original = _recorder(fail_on_mps=True)
    x = torch.randn(256, device="mps")

    out = patch._requant(original, x, torch.float8_e4m3fn, 0)

    assert original.seen == ["mps", "cpu"], f"expected a CPU retry: {original.seen}"
    assert out.device.type == "mps", "result must come back to the caller's device"
    assert out.dtype is torch.float8_e4m3fn


@requires_mps
def test_a_failed_native_path_is_not_retried_per_weight():
    """A LoRA re-quantises every weight it touches. Re-raising and re-catching on
    each one would put the exception cost back on the hot path #29 is about."""
    original = _recorder(fail_on_mps=True)
    x = torch.randn(256, device="mps")

    for _ in range(5):
        patch._requant(original, x, torch.float8_e4m3fn, 0)

    assert original.seen.count("mps") == 1, (
        f"native path retried {original.seen.count('mps')}x across 5 weights"
    )
    assert original.seen.count("cpu") == 5


def test_non_fp8_targets_are_passed_straight_through():
    original = _recorder()
    x = torch.randn(16)

    patch._requant(original, x, torch.bfloat16, 0)

    assert original.seen == ["cpu"]


@requires_mps
def test_native_requant_is_bit_exact_vs_the_cpu_round_trip():
    """The anchor for the whole change: moving the maths back onto the GPU must
    not move the numbers. Compared byte-for-byte in the stored fp8 encoding."""
    import comfy_kitchen.backends.eager.quantization as q
    from _patches import tensor_to_fp8

    tensor_to_fp8.install()   # supplies the float->fp8 cast the GPU path needs

    rng = torch.rand(8192, dtype=torch.float16)

    def original(value, dtype, seed=0):
        return q.stochastic_rounding_fp8(value, rng.to(value.device), dtype)

    x = (torch.randn(8192) * 3).to("mps")

    native = patch._requant(original, x, torch.float8_e4m3fn, 0)
    reference = original(x.cpu(), torch.float8_e4m3fn).to("mps")

    assert torch.equal(
        native.cpu().view(torch.uint8), reference.cpu().view(torch.uint8)
    ), "GPU re-quant differs from the CPU round-trip it replaces"
