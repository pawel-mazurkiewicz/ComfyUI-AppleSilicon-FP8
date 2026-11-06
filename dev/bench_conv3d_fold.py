# dev/bench_conv3d_fold.py
"""Time stock conv3d, im2col and the multi-frame tap-fold on video-VAE conv3d shapes.

Not run by the test suite. im2col is measured only where matmul2d exists on this Mac.
`fold_1f` forces one output frame per conv2d (tight budget) to price the frame batching.
"""
import os
import sys
import time

import torch

# repo root (parent of dev/) on path so `_patches` imports when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _patches import _caps  # noqa: E402
from _patches import conv_im2col_mps as cm  # noqa: E402

DT = torch.bfloat16
STRIDE, PADDING, DILATION = 1, (1, 1, 1), 1
WARM, ITERS = 10, 10
# (C, T, H, W) with Co = C, kT = 3, kH = kW = 3: Wan / LTX VAE decoder conv shapes
SHAPES = [(384, 6, 60, 104), (192, 6, 120, 208), (512, 8, 32, 32)]


def _conv3d(*args):
    return (cm._orig_conv3d or torch.nn.functional.conv3d)(*args)


def _fold(x, w, b, mb=None):
    if mb is None:
        return cm._folded_conv3d(x, w, b, STRIDE, PADDING, DILATION, 1)
    prev = os.environ.get("ASFP8_CONV_FOLD_MB")
    os.environ["ASFP8_CONV_FOLD_MB"] = str(mb)
    try:
        return cm._folded_conv3d(x, w, b, STRIDE, PADDING, DILATION, 1)
    finally:
        if prev is None:
            os.environ.pop("ASFP8_CONV_FOLD_MB", None)
        else:
            os.environ["ASFP8_CONV_FOLD_MB"] = prev


def _cands(x, w, b):
    out = [
        ("stock", lambda: _conv3d(x, w, b, STRIDE, PADDING, DILATION)),
        ("fold", lambda: _fold(x, w, b)),
        ("fold_1f", lambda: _fold(x, w, b, "0.01")),   # one output frame per conv2d
    ]
    if _caps.has_tensor_ops_matmul2d() and cm._supported(x, w, DILATION, 1, DT):
        out.insert(1, ("im2col", lambda: cm.conv_im2col(x, w, b, STRIDE, PADDING, DILATION, 1)))
    return out


def _timed(fn):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _rel_rms(out, ref):
    return ((out.float().cpu() - ref).norm() / ref.norm()).item()


def correctness():
    """fp32-CPU gate on a small proxy of the benched config, before any timing."""
    C, T, HW = 32, 6, 24
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, C, T, HW, HW, generator=g).to(DT).to("mps")
    w = (torch.randn(C, C, 3, 3, 3, generator=g) / (C * 27) ** 0.5).to(DT).to("mps")
    b = torch.randn(C, generator=g).to(DT).to("mps")
    ref = torch.nn.functional.conv3d(x.float().cpu(), w.float().cpu(), b.float().cpu(),
                                      STRIDE, PADDING, DILATION)
    fold = _fold(x, w, b)
    e_fold, e_stock = _rel_rms(fold, ref), _rel_rms(_conv3d(x, w, b, STRIDE, PADDING, DILATION), ref)
    ok = e_fold <= max(e_stock * 1.05, 1e-6)
    print(f"correctness  fold_rel_rms={e_fold:.3g}  stock_rel_rms={e_stock:.3g}  -> "
          f"{'OK' if ok else 'FAIL'}\n")
    if not ok:
        raise SystemExit(1)


def bench(shape):
    C, T, H, W = shape
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, C, T, H, W, generator=g).to(DT).to("mps")
    w = (torch.randn(C, C, 3, 3, 3, generator=g) / (C * 27) ** 0.5).to(DT).to("mps")
    b = torch.randn(C, generator=g).to(DT).to("mps")
    cands = _cands(x, w, b)
    for _ in range(WARM):
        for _, fn in cands:
            fn()
    acc = {n: 0.0 for n, _ in cands}
    for _ in range(ITERS):                       # interleaved, so thermal drift is shared
        for n, fn in cands:
            acc[n] += _timed(fn)
    ms = {n: acc[n] / ITERS for n, _ in cands}
    gather_frame_mb = (1 * C * 3 * H * W * x.element_size()) / 1024 ** 2
    n_chunk = cm._fold_chunk_frames(1, C, 3, H, W, x.element_size(), cm._fold_budget_bytes())
    print(f"C={C} T={T} {H}x{W}  bf16 kT=3 pad={PADDING}  "
          f"gather={gather_frame_mb:.0f}MB/frame  frames/conv2d={min(n_chunk, T)}")
    for n, _ in cands:
        print(f"  {n:<8} {ms[n]:8.3f} ms   {ms['stock'] / ms[n]:5.2f}x vs stock")
    del x, w, b
    torch.mps.empty_cache()


def main():
    if not torch.backends.mps.is_available():
        print("MPS not available; nothing to bench.")
        return
    correctness()
    for shape in SHAPES:
        bench(shape)
    print(f"\nwarm={WARM} timed={ITERS} interleaved; budget ASFP8_CONV_FOLD_MB="
          f"{os.environ.get('ASFP8_CONV_FOLD_MB', '512')}MiB")


if __name__ == "__main__":
    main()
