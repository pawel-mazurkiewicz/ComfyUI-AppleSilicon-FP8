import pytest
import torch

from conftest import requires_mps

from _patches import comfykitchen_fp8 as ck


# --- wrapper logic (no comfy_kitchen needed) ---


def test_passthrough_when_not_mps():
    calls = {}

    def fake_orig(qx, scale, output_type=torch.bfloat16):
        calls["dev"] = qx.device.type
        return qx.float()

    wrapped = ck._cpu_dequant_on_mps(fake_orig)
    out = wrapped(torch.zeros(2, 2), torch.ones(1))
    assert calls["dev"] == "cpu"          # orig saw the args unchanged
    assert out.device.type == "cpu"


@requires_mps
def test_reroute_runs_on_cpu_and_returns_to_device():
    seen = {}

    def fake_orig(qx, scale, block_scales, output_type=torch.bfloat16):
        # The whole point: every tensor must arrive on CPU (MPS can't do the fp8 work).
        seen["devs"] = (qx.device.type, scale.device.type, block_scales.device.type)
        return torch.ones(2, 2, dtype=torch.bfloat16)   # CPU float result

    wrapped = ck._cpu_dequant_on_mps(fake_orig)
    qx = torch.zeros(2, 2, dtype=torch.float8_e4m3fn, device="mps")
    out = wrapped(qx, torch.ones(1, device="mps"), torch.zeros(2, 2, dtype=torch.float8_e4m3fn, device="mps"))

    assert seen["devs"] == ("cpu", "cpu", "cpu")
    assert out.device.type == "mps"       # float result moved back to the device


# --- real comfy_kitchen NVFP4 parity on MPS (gated: needs comfy_kitchen + MPS) ---

try:
    import comfy_kitchen.backends.eager.quantization as _q  # noqa: F401
    _HAS_CK = True
except Exception:
    _HAS_CK = False

requires_ck = pytest.mark.skipif(not _HAS_CK, reason="comfy_kitchen not installed in this env")


@requires_mps
@requires_ck
def test_dequantize_nvfp4_mps_matches_cpu():
    import comfy_kitchen.backends.eager.quantization as q

    ck.install()  # idempotent enough for one call; wraps dequantize_nvfp4 on MPS

    torch.manual_seed(0)
    x = torch.randn(256, 256) * 0.3
    pts = torch.tensor(float(x.abs().max()) / (6.0 * 448.0) + 1e-8)
    qx, block_scales = q.quantize_nvfp4(x, pts)             # valid NVFP4 inputs (CPU)

    ref = q.dequantize_nvfp4(qx, pts, block_scales, torch.bfloat16)
    out = q.dequantize_nvfp4(qx.to("mps"), pts.to("mps"), block_scales.to("mps"), torch.bfloat16)

    assert out.device.type == "mps"
    rel = ((out.cpu().float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-9)).item()
    assert rel < 1e-3, f"NVFP4 MPS dequant rel error {rel:.4e}"
