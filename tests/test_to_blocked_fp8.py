"""Issue #8: loading an NVFP4 checkpoint on MPS dies in comfy.float.to_blocked.

It pads the block-scale matrix with `padded[:rows, :cols] = input_matrix`, a strided fp8
copy MPS has no kernel for. fp8_mps_strided wraps reshape/contiguous/clone, so it covers
the later swizzle but not this __setitem__.
"""
import os
import sys

import pytest
import torch

from conftest import requires_mps

_CANDIDATES = [
    os.environ.get("ASFP8_COMFY_PATH"),
    "/Users/pawelma/ComfyUI-Installs/ComfyUI/ComfyUI",
]
for _c in _CANDIDATES:
    if _c and os.path.isdir(os.path.join(_c, "comfy")) and _c not in sys.path:
        sys.path.insert(0, _c)

cf = pytest.importorskip("comfy.float")

from _patches import stochastic_round_fp8 as sr  # noqa: E402

# (out_features, in_features // 16); the broken case is column padding, in_features % 64
SHAPES = [
    (130, 6),    # rows and cols padded
    (128, 6),    # cols only  -- the strided setitem
    (100, 4),    # rows only
    (128, 4),    # no padding -- swizzle path only
    (1, 1),
]


def _bits(t):
    return t.cpu().view(torch.uint8)


@requires_mps
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("flatten", [False, True])
def test_to_blocked_fp8_on_mps_matches_cpu(shape, flatten):
    sr.install()

    rows, cols = shape
    ref_in = torch.arange(rows * cols, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(rows, cols)
    ref = cf.to_blocked(ref_in, flatten=flatten)

    out = cf.to_blocked(ref_in.to("mps"), flatten=flatten)

    assert out.device.type == "mps"
    assert out.dtype == torch.float8_e4m3fn
    assert out.shape == ref.shape
    assert torch.equal(_bits(out), _bits(ref))


@requires_mps
def test_to_blocked_uint8_unaffected():
    """uint8 E8M0 scales are left alone: MPS handles those natively."""
    sr.install()

    x = torch.arange(130 * 6, dtype=torch.uint8).reshape(130, 6)
    ref = cf.to_blocked(x, flatten=False)
    out = cf.to_blocked(x.to("mps"), flatten=False)

    assert out.device.type == "mps"
    assert torch.equal(out.cpu(), ref)


@requires_mps
def test_nvfp4_quantize_block_scales_match_cpu():
    """NVFP4 quantize block-scales match the CPU bit-for-bit.

    Only the scales: the fp4 payload uses a device RNG and can't cross devices.
    """
    from _patches import fp8_mps_strided, tensor_to_fp8
    fp8_mps_strided.install()
    tensor_to_fp8.install()
    sr.install()

    torch.manual_seed(0)
    w = torch.randn(128, 96) * 0.3          # 96 // 16 = 6 block-scale columns -> needs padding
    scale = torch.tensor(float(w.abs().max()) / (6.0 * 448.0) + 1e-8)

    _, ref_scales = cf.stochastic_round_quantize_nvfp4_by_block(w, scale, pad_16x=False, seed=7)
    _, mps_scales = cf.stochastic_round_quantize_nvfp4_by_block(
        w.to("mps"), scale.to("mps"), pad_16x=False, seed=7
    )

    assert mps_scales.device.type == "mps"
    assert torch.equal(_bits(mps_scales), _bits(ref_scales))


# --- the same bug in comfy_kitchen.float_utils.to_blocked (identical implementation) ---

try:
    import comfy_kitchen as _ck  # noqa: F401
    _HAS_CK = True
except ImportError:
    _HAS_CK = False

requires_ck = pytest.mark.skipif(not _HAS_CK, reason="comfy_kitchen not installed in this env")


@requires_mps
@requires_ck
def test_ck_quantize_nvfp4_on_mps_matches_cpu():
    """comfy_kitchen's own NVFP4 quantize matches the CPU; it defaults to non-stochastic."""
    import comfy_kitchen as ck

    from _patches import comfykitchen_fp8, fp8_mps_strided, tensor_to_fp8
    fp8_mps_strided.install()
    tensor_to_fp8.install()
    comfykitchen_fp8.install()

    torch.manual_seed(0)
    w = torch.randn(128, 96) * 0.3
    scale = torch.tensor(float(w.abs().max()) / (6.0 * 448.0) + 1e-8)

    ref_q, ref_scales = ck.quantize_nvfp4(w, scale, pad_16x=False)
    mps_q, mps_scales = ck.quantize_nvfp4(w.to("mps"), scale.to("mps"), pad_16x=False)

    assert mps_scales.device.type == "mps"
    assert torch.equal(_bits(mps_scales), _bits(ref_scales))
    assert torch.equal(mps_q.cpu().view(torch.uint8), ref_q.view(torch.uint8))
