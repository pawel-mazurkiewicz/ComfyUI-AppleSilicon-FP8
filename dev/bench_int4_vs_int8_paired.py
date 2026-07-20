"""Paired int4-vs-int8 bench: both models in ONE process, evals INTERLEAVED.

Why: separately-launched runs of the same model disagreed by 13% (4.135 vs 3.651
s/eval), and a single 8-eval run drifts ~5% monotonically (thermal). Both are
larger than the int4-vs-int8 effect being measured, so sequential process runs
cannot resolve it. Interleaving A/B/A/B makes thermal state and drift common-mode
to both arms; the paired per-round ratio is then meaningful even if the absolute
numbers wander.

Also sweeps M (latent size), because the harness (M=4096) and the live ComfyUI
workflow disagree about the int4/int8 ratio (1.09x vs 1.95x) and M is one of the
few things that differs.

Run:
  ASFP8_INT4_EXT=1 ASFP8_INT8_EXT=1 "<venv>/bin/python" dev/bench_int4_vs_int8_paired.py [rounds]
"""

import os
import statistics
import sys
import time

COMFY_MASTER = os.path.expanduser(os.environ.get("ASFP8_COMFY", "~/ComfyUI-Installs/ComfyUI/ComfyUI"))
NODE_DIR = "/Volumes/IMPERIAL SPACE/AI/ComfyUI/custom_nodes/ComfyUI-AppleSilicon-FP8"
MODELS = "/Volumes/IMPERIAL SPACE/AI/ComfyUI/models/diffusion_models"

sys.path.insert(0, COMFY_MASTER)
sys.path.insert(0, NODE_DIR)

from _patches import psutil_vmstat  # noqa: E402

psutil_vmstat.install()

import torch  # noqa: E402

import comfy.sd  # noqa: E402
import comfy.model_management as mm  # noqa: E402

from _patches import int4_linear_mps, int8_linear_kernel_mps, int8_linear_mps, int_mm_mps  # noqa: E402

for _p in (int4_linear_mps, int_mm_mps, int8_linear_mps, int8_linear_kernel_mps):
    _p.install()

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

ARMS = [
    ("int4", os.path.join(MODELS, "krea2_turbo-int4_convrot.safetensors")),
    ("int8", os.path.join(MODELS, "Krea2_Turbo_convrot_int8mixed.safetensors")),
]

loaded = []
for name, path in ARMS:
    print(f"loading {name} ...", flush=True)
    m = comfy.sd.load_diffusion_model(path)
    mm.load_models_gpu([m], force_full_load=True)
    loaded.append((name, m, m.model.diffusion_model))

device = mm.get_torch_device()


def make_inputs(dm, dtype, hw):
    c_in = (dm.input_embed.proj.in_features // (dm.input_embed.patch ** 2)
            if hasattr(dm, "input_embed") else 16)
    x = torch.randn(1, c_in, hw, hw, device=device, dtype=dtype)
    context = torch.randn(1, 512, 12 * 2560, device=device, dtype=dtype)
    ts = torch.full((1,), 0.5, device=device, dtype=dtype)
    return x, ts, context


def one_eval(dm, inputs):
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = dm(*inputs)
        torch.mps.synchronize()
        return time.perf_counter() - t0


for hw in (96, 128, 160):
    tokens = (hw // 2) ** 2
    print(f"\n{'=' * 62}\nlatent {hw}x{hw}  (~{hw * 8}px, M={tokens} tokens)\n{'=' * 62}")

    prepared = []
    for name, m, dm in loaded:
        inp = make_inputs(dm, m.model.get_dtype(), hw)
        for _ in range(2):  # warmup this arm at this shape
            one_eval(dm, inp)
        prepared.append((name, dm, inp))

    series = {n: [] for n, _, _ in prepared}
    ratios = []
    for r in range(ROUNDS):
        got = {}
        for name, dm, inp in prepared:  # interleaved within each round
            t = one_eval(dm, inp)
            series[name].append(t)
            got[name] = t
        ratios.append(got["int4"] / got["int8"])

    for name in series:
        s = sorted(series[name])
        med = s[len(s) // 2]
        spread = (s[-1] - s[0]) / med * 100
        print(f"  {name}: median {med:.3f}s  (min {s[0]:.3f}, max {s[-1]:.3f}, spread {spread:.1f}%)")

    med_r = statistics.median(ratios)
    print(f"  paired int4/int8 ratio: median {med_r:.3f}  "
          f"(min {min(ratios):.3f}, max {max(ratios):.3f}, n={ROUNDS})")
    print(f"  -> int4 is {abs(1 - med_r) * 100:.1f}% {'SLOWER' if med_r > 1 else 'FASTER'} than int8")
