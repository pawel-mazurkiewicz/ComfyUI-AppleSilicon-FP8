"""Issue-F bench: native fp8 NT vs LUT->bf16, on Flux shapes. Verifies before timing.
Reports: native(kernel-only), native(end-to-end incl bf16->half cast),
         lut(GEMM-only, weight pre-decoded), lut(end-to-end incl per-call decode_fp8)."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ASFP8_FP8_NATIVE"] = "1"
from _patches._common import decode_fp8
from _patches.fp8_ext import loader

mod = loader.module(); assert mod is not None; mod.warmup()
dev = "mps"
SHAPES = [(4096, 4096, 4096), (4096, 4096, 16384), (4096, 16384, 4096)]
ITERS = 30

def sync(): torch.mps.synchronize()
def timed(fn):
    for _ in range(5): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(ITERS): fn()
    sync(); return (time.perf_counter() - t0) / ITERS * 1e3

for (M, K, N) in SHAPES:
    g = torch.Generator().manual_seed(1)
    act = (torch.randn(M, K, generator=g) * 0.3).to(torch.bfloat16).to(dev)
    w   = (torch.randn(N, K, generator=g) * 0.3).to(torch.float8_e4m3fn).to(dev)
    w_u8_pre = w.contiguous().view(torch.uint8)
    a_half_pre = act.to(torch.float16).contiguous()

    native_kernel_only = lambda: mod.fp8_matmul2d_nt(a_half_pre, w_u8_pre, N)
    # end-to-end: cast happens every call, as in _fp8_linear_kernel
    native_e2e = lambda: mod.fp8_matmul2d_nt(act.to(torch.float16).contiguous(), w_u8_pre, N)
    w_bf16_pre = decode_fp8(w, torch.bfloat16)
    lut_gemm_only = lambda: (act @ w_bf16_pre.t())
    lut_e2e = lambda: (act @ decode_fp8(w, torch.bfloat16).t())  # decode every call (real model)

    # VERIFY FIRST (gate the whole bench).
    gt = (act.float() @ decode_fp8(w, torch.float32).t())
    rel_n = ((native_kernel_only().float() - gt).abs().max() / (gt.abs().max() + 1e-9)).item()
    assert rel_n < 2e-2, f"native rel={rel_n} too high; refusing to time"

    tk  = timed(native_kernel_only)
    tne = timed(native_e2e)
    tlg = timed(lut_gemm_only)
    tle = timed(lut_e2e)
    print(f"M={M} K={K} N={N}: native_kernel={tk:.3f}ms native_e2e={tne:.3f}ms "
          f"lut_gemm={tlg:.3f}ms lut_e2e(+decode)={tle:.3f}ms "
          f"speedup_e2e={tle/tne:.2f}x  rel={rel_n:.2e}")
# Honest comparison: the model-relevant number is native_e2e vs lut_e2e (both include their
# real per-call costs). native_kernel vs lut_gemm is the kernel-only ceiling.
