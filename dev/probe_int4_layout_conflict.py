"""Does ComfyUI-INT4-Fast's stub layout registration change the int4 path?

Real ComfyUI logs "Int4Fast: Registered TensorCoreConvRotW4A4Layout dynamically",
which only fires when the key is ABSENT from comfy.quant_ops.QUANT_ALGOS — i.e.
INT4-Fast inserts its own empty `class ...(QuantizedLayout): pass` stub. The
headless harness never imports custom nodes, so it sees whatever comfy/kitchen
registers. That is a real difference between harness (int4/int8 = 1.09x) and live
ComfyUI (1.95x).

This prints who owns each layout key before/after importing INT4-Fast, then
benches the model under the requested condition.

  ASFP8_LOAD_INT4FAST=1  import ComfyUI-INT4-Fast first (simulate live ComfyUI)

Run:
  "<venv>/bin/python" dev/probe_int4_layout_conflict.py <ckpt> [evals]
"""

import importlib.util
import os
import sys
import time

COMFY_MASTER = os.path.expanduser(os.environ.get("ASFP8_COMFY", "~/ComfyUI-Installs/ComfyUI/ComfyUI"))
NODE_DIR = "/Volumes/IMPERIAL SPACE/AI/ComfyUI/custom_nodes/ComfyUI-AppleSilicon-FP8"
INT4FAST_DIR = "/Volumes/IMPERIAL SPACE/AI/ComfyUI/custom_nodes/ComfyUI-INT4-Fast"

sys.path.insert(0, COMFY_MASTER)
sys.path.insert(0, NODE_DIR)

from _patches import psutil_vmstat  # noqa: E402

psutil_vmstat.install()

import torch  # noqa: E402

import comfy.sd  # noqa: E402
import comfy.model_management as mm  # noqa: E402


def dump_algos(tag):
    from comfy.quant_ops import QUANT_ALGOS
    print(f"\n--- QUANT_ALGOS [{tag}] ---")
    for k in sorted(QUANT_ALGOS):
        if "ConvRot" in k or "convrot" in k or "INT8" in k:
            cls = QUANT_ALGOS[k]
            mod = getattr(cls, "__module__", "?")
            # a stub has no meaningful body: no Params, no ops
            has_params = hasattr(cls, "Params")
            print(f"  {k:38s} <- {mod}  (Params={has_params})")


dump_algos("before custom nodes")

if os.environ.get("ASFP8_LOAD_INT4FAST") == "1":
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_INT4_Fast", os.path.join(INT4FAST_DIR, "__init__.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_INT4_Fast"] = m
    sys.path.insert(0, os.path.dirname(INT4FAST_DIR))
    spec.loader.exec_module(m)
    print("\n[probe] imported ComfyUI-INT4-Fast")
    dump_algos("after INT4-Fast import")

from _patches import int4_linear_mps  # noqa: E402

int4_linear_mps.install()

# count how many times our patched dispatcher actually runs
_hits = {"n": 0}
_patched_fn = None
import comfy_kitchen.tensor.convrot_w4a4 as ck_convrot  # noqa: E402

_patched_fn = ck_convrot.convrot_w4a4_linear


def _counting(*a, **kw):
    _hits["n"] += 1
    return _patched_fn(*a, **kw)


ck_convrot.convrot_w4a4_linear = _counting

ckpt = sys.argv[1]
n_evals = int(sys.argv[2]) if len(sys.argv) > 2 else 5

model = comfy.sd.load_diffusion_model(ckpt)
mm.load_models_gpu([model], force_full_load=True)
dm = model.model.diffusion_model

sample = None
n_quant = 0
for name, mod in dm.named_modules():
    w = getattr(mod, "weight", None)
    if w is not None and w.__class__.__name__ == "QuantizedTensor":
        n_quant += 1
        if sample is None:
            sample = (name, type(w._params).__module__ + "." + type(w._params).__qualname__)
print(f"\nquantized layers: {n_quant}")
print(f"sample params class: {sample}")

device = mm.get_torch_device()
dtype = model.model.get_dtype()
c_in = (dm.input_embed.proj.in_features // (dm.input_embed.patch ** 2)
        if hasattr(dm, "input_embed") else 16)
x = torch.randn(1, c_in, 128, 128, device=device, dtype=dtype)
context = torch.randn(1, 512, 12 * 2560, device=device, dtype=dtype)
ts = torch.full((1,), 0.5, device=device, dtype=dtype)

with torch.no_grad():
    for _ in range(2):
        _ = dm(x, ts, context)
        torch.mps.synchronize()
    hits_before = _hits["n"]
    times = []
    for i in range(n_evals):
        t0 = time.perf_counter()
        _ = dm(x, ts, context)
        torch.mps.synchronize()
        times.append(time.perf_counter() - t0)

per_eval = (_hits["n"] - hits_before) / n_evals
times = sorted(times)
print(f"\nconvrot_w4a4_linear calls/eval: {per_eval:.0f}  (0 => our patch is BYPASSED)")
print(f"median {times[len(times) // 2]:.3f} s/eval (min {times[0]:.3f}, n={n_evals})")
