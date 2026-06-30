"""Issue-F Task 0 probe (run on M5, ASFP8_FP8_NATIVE=1).

SYNTHETIC ONLY. Confirms the existing native fp8 NT kernel (half activation x fp8
e4m3 weight) matches the decoded-fp32 ground truth on real Flux-2-Klein linear
shapes, and that the current LUT->bf16 path is not dramatically more accurate.
Cannot establish real activation range (see Task -1). Lifts the layout/spy
approach from docs/superpowers/results/G2-results.md.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _patches._common import decode_fp8
from _patches.fp8_ext import loader

assert torch.backends.mps.is_available(), "need MPS"
os.environ["ASFP8_FP8_NATIVE"] = "1"
mod = loader.module()
assert mod is not None, "fp8_ext failed to build"
mod.warmup()
assert hasattr(mod, "fp8_matmul2d_nt"), "fp8_matmul2d_nt export missing"

# Flux-2-Klein-9B-class linear shapes: M=tokens(1024^2 -> ~4096), K/N in {4096,16384}.
SHAPES = [(4096, 4096, 4096), (4096, 4096, 16384), (4096, 16384, 4096)]
g = torch.Generator().manual_seed(0)
worst = 0.0
ok = True
for (M, K, N) in SHAPES:
    act_bf16 = (torch.randn(M, K, generator=g) * 0.3).to(torch.bfloat16)        # activation
    w_fp8    = (torch.randn(N, K, generator=g) * 0.3).to(torch.float8_e4m3fn)   # weight [N,K]

    # Ground truth: decode both to fp32, fp32 matmul.
    gt = (act_bf16.float() @ decode_fp8(w_fp8.to("mps"), torch.float32).cpu().t()).to("mps")

    # NATIVE: half activation x fp8 weight bytes.
    a_half = act_bf16.to("mps").to(torch.float16).contiguous()
    w_u8   = w_fp8.to("mps").contiguous().view(torch.uint8)
    native = mod.fp8_matmul2d_nt(a_half, w_u8, N)            # [M,N] f32

    # CURRENT (LUT->bf16): decode weight to bf16, bf16 matmul (what we replace).
    a_bf16 = act_bf16.to("mps")
    w_bf16 = decode_fp8(w_fp8.to("mps"), torch.bfloat16)
    lut    = (a_bf16 @ w_bf16.t()).float()

    assert native.abs().max() > 0, "SPY: native C is all-zero -> kernel no-op"
    diff = (native - gt).abs()
    den  = gt.abs().max() + 1e-9
    rel_native = (diff.max() / den).item()
    rel_lut    = ((lut - gt).abs().max() / den).item()
    # Distribution-aware stats (MAJOR 10): worst-element relative is not enough over K<=16384.
    rel_elt = (diff / (gt.abs() + 1e-6))
    mean_abs = diff.mean().item()
    p999     = torch.quantile(rel_elt.flatten().float()[:: max(1, rel_elt.numel()//1_000_000)], 0.999).item()
    actmax   = act_bf16.float().abs().max().item()
    worst = max(worst, rel_native)
    # Decision rule (MAJOR 10): native must be within ~2x of the LUT path it replaces.
    within_lut = rel_native <= 2 * rel_lut + 1e-4
    ok = ok and (rel_native < 2e-2) and within_lut
    print(f"M={M} K={K} N={N}: rel_native={rel_native:.4e} rel_lut={rel_lut:.4e} "
          f"mean_abs={mean_abs:.4e} p99.9_rel={p999:.4e} act|max|={actmax:.3f} "
          f"within_2x_lut={within_lut} (fp16 max 65504)")

print(f"WORST rel_native={worst:.4e}  (gate: < 2e-2 AND rel_native <= 2*rel_lut+eps)")
assert ok, "native path failed the tolerance/decision gate vs fp32 ground truth"
print("PROBE PASS (synthetic; real range/scale still gated on Task -1)")
