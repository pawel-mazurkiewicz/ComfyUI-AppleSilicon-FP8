"""Smoke + micro-bench for ConvRot W4A4 (int4) on MPS via comfy master + kitchen 0.2.19.

Run with the ComfyUI venv python:
  "/Volumes/IMPERIAL SPACE/AI/ComfyUI/.venv/bin/python" dev/smoke_int4_convrot.py

Injects the master worktree + isolated kitchen 0.2.19 (does NOT touch the live install).

Measures per-Linear-call cost on Ideogram/Krea-shaped GEMMs:
  1. plain bf16 F.linear                        (upper bound / reference)
  2. int8-style: dequant int8 weight -> F.linear (what int8 models do w/ our patch)
  3. eager convrot_w4a4 dispatch (F.linear on QuantizedTensor) = what INT4 models do today
  4. proposed quick fix: rotate act (grouped GEMM) + W4 unpack->bf16 + F.linear (no act round-trip)
plus correctness of (3)/(4) vs reference.
"""

import os
import sys
import time

CK_TARGET = os.environ.get("ASFP8_CK_TARGET", "")  # venv kitchen is new enough now
COMFY_MASTER = os.path.expanduser(os.environ.get("ASFP8_COMFY", "~/ComfyUI-Installs/ComfyUI/ComfyUI"))
NODE_DIR = "/Volumes/IMPERIAL SPACE/AI/ComfyUI/custom_nodes/ComfyUI-AppleSilicon-FP8"

if CK_TARGET:
    sys.path.insert(0, CK_TARGET)
sys.path.insert(0, COMFY_MASTER)
sys.path.insert(0, NODE_DIR)

from _patches import psutil_vmstat  # noqa: E402

psutil_vmstat.install()

import torch  # noqa: E402

import comfy_kitchen  # noqa: E402
import comfy.quant_ops as q  # noqa: E402

print("kitchen:", comfy_kitchen.__version__, os.path.dirname(comfy_kitchen.__file__))
print("layout present:", hasattr(q, "TensorCoreConvRotW4A4Layout"))

from comfy_kitchen.registry import registry  # noqa: E402

impl = registry.get_implementation("convrot_w4a4_linear", kwargs=None)
print("convrot_w4a4_linear impl:", impl.__module__)

dev = "mps"
torch.manual_seed(0)

M, K, N = 4096, 4608, 4608  # Ideogram/Krea-shaped
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02

Layout = q.TensorCoreConvRotW4A4Layout
qdata, params = Layout.quantize(w.float(), convrot_groupsize=256, quant_group_size=64,
                                stochastic_rounding=0, linear_dtype="int4")
qt = q.QuantizedTensor(qdata, "TensorCoreConvRotW4A4Layout", params)
print("qdata:", qdata.dtype, tuple(qdata.shape), "scale:", params.scale.dtype, tuple(params.scale.shape))

# int8 comparison weight (per-row symmetric, like int8-fast dequant path)
w_absmax = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
w8_scale = w_absmax / 127.0
w8 = (w.float() / w8_scale).round().clamp(-127, 127).to(torch.int8)

ref = torch.nn.functional.linear(x.float(), w.float())

# --- correctness: eager convrot dispatch
y_convrot = torch.nn.functional.linear(x, qt)
err_convrot = (y_convrot.float() - ref).abs().mean() / ref.abs().mean()

# --- proposed quick fix: keep rotated basis, no activation quant round-trip
from comfy_kitchen.backends.eager.convrot_w4a4 import (  # noqa: E402
    _build_hadamard, _rotate_activation, _unpack_int4_row_major)


def w4a16_rotated(x, qdata, scale, groupsize=256):
    h = _build_hadamard(groupsize, device=x.device, dtype=x.dtype)
    x_rot = _rotate_activation(x, h, groupsize)
    w_rot = _unpack_int4_row_major(qdata).to(x.dtype) * scale.to(x.device, x.dtype).reshape(-1, 1)
    return torch.nn.functional.linear(x_rot, w_rot)


y_fix = w4a16_rotated(x, qdata.to(dev), params.scale)
err_fix = (y_fix.float() - ref).abs().mean() / ref.abs().mean()

print(f"rel err  eager convrot W4A4: {err_convrot.item():.4%}")
print(f"rel err  quickfix W4A16    : {err_fix.item():.4%}")


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


qdata_mps = qdata.to(dev)
scale_mps = params.scale.to(dev)
qt_mps = q.QuantizedTensor(qdata_mps, "TensorCoreConvRotW4A4Layout",
                           type(params)(scale=scale_mps, orig_dtype=params.orig_dtype,
                                        orig_shape=params.orig_shape,
                                        convrot_groupsize=params.convrot_groupsize,
                                        quant_group_size=params.quant_group_size,
                                        linear_dtype=params.linear_dtype))

t_bf16 = bench(lambda: torch.nn.functional.linear(x, w))
t_int8 = bench(lambda: torch.nn.functional.linear(x, (w8.to(torch.bfloat16) * w8_scale.to(torch.bfloat16))))
t_convrot = bench(lambda: torch.nn.functional.linear(x, qt_mps))
t_fix = bench(lambda: w4a16_rotated(x, qdata_mps, scale_mps))

print(f"\nper-call ms @ M={M} K={K} N={N}")
print(f"  bf16 F.linear            : {t_bf16:7.3f}")
print(f"  int8 dequant + F.linear  : {t_int8:7.3f}")
print(f"  eager convrot W4A4 (now) : {t_convrot:7.3f}   ({t_convrot / t_int8:.2f}x vs int8)")
print(f"  quickfix W4A16 (proposed): {t_fix:7.3f}   ({t_fix / t_int8:.2f}x vs int8)")
