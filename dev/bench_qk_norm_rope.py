"""Bench: eager comfy_kitchen rms_rope vs the fused QK-norm+RoPE kernel (Z-Image shapes).

    python dev/bench_qk_norm_rope.py [S]

Prints per-block timings (q+k, sync-timed medians) and rel-RMS against an fp32 reference.
Exits non-zero if the fused output is not within 1.05x of eager's error or its layout
differs from eager's.
"""
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if os.environ.get("COMFYUI_ROOT"):
    sys.path.insert(0, os.environ["COMFYUI_ROOT"])

import torch
import torch.nn.functional as F
import comfy_kitchen as ck
from _patches import rope_fast_mps, qk_norm_rope_mps as qn

dev, bf = "mps", torch.bfloat16
B, H, D = 1, 30, 128
S = int(sys.argv[1]) if len(sys.argv) > 1 else 4160
EPS = 1e-5


def bench(fn, n=30, w=5):
    for _ in range(w):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def rel(a, b):
    a, b = a.float().cpu(), b.float().cpu()
    return ((a - b).norm() / b.norm()).item()


torch.manual_seed(0)
th = torch.rand(S, D // 2, device=dev) * 3
c, s = th.cos(), th.sin()
fz = torch.stack([c, -s, s, c], -1).view(1, S, 1, D // 2, 2, 2)
qkv = (torch.randn(B, S, 3 * H * D, device=dev) * 3).to(bf)
w = (torch.rand(D, device=dev) + 0.5).to(bf)
xq, xk, _ = (t.view(B, S, H, D) for t in torch.split(qkv, [H * D] * 3, dim=-1))


def ref32(x):
    x = x.float()
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
    xr = n.reshape(B, S, H, D // 2, 1, 2)
    return (fz[..., 0] * xr[..., 0] + fz[..., 1] * xr[..., 1]).reshape(B, S, H, D)


def chain(contig=False, attn=False):
    def run():
        q, k = ck.rms_rope(xq, xk, fz, w, w, EPS)
        q, k = q.movedim(1, 2), k.movedim(1, 2)
        if contig:
            q, k = q.contiguous(), k.contiguous()
        if attn:
            return F.scaled_dot_product_attention(q, k, k)
        return q, k
    return run


rope_fast_mps.install_for_test()
eager = chain()()
t_e = [bench(chain()), bench(chain(contig=True)), bench(chain(attn=True), n=10, w=2)]
qn.install_for_test()
fused = chain()()
assert qn._last_backend() == "kernel", "fused path fell back"
t_f = [bench(chain()), bench(chain(contig=True)), bench(chain(attn=True), n=10, w=2)]

ok = True
for nm, x, e, f in (("q", xq, eager[0], fused[0]), ("k", xk, eager[1], fused[1])):
    r = ref32(x).movedim(1, 2)
    re, rf = rel(e, r), rel(f, r)
    same = f.shape == e.shape and f.stride() == e.stride() and f.dtype == e.dtype
    ok &= rf <= 1.05 * re and same
    print(f"{nm}: rel-RMS vs fp32 eager {re:.3e} fused {rf:.3e} ratio {rf / re:.3f}; "
          f"strides eager {e.stride()} fused {f.stride()}")
print("rms_rope+movedim ms          eager %.3f fused %.3f" % (t_e[0], t_f[0]))
print("... + contiguous ms          eager %.3f fused %.3f" % (t_e[1], t_f[1]))
print("... + SDPA(q,k,k) ms         eager %.3f fused %.3f" % (t_e[2], t_f[2]))
qn.uninstall_for_test()
rope_fast_mps.uninstall_for_test()
sys.exit(0 if ok else 1)
