# dev/bench_mps_conv.py
"""Baseline stock MPS conv2d/conv3d vs the MEASURED matmul GEMM at the same M,N,K.
Decides whether conv2d is already competitive (=> only conv3d needs the im2col kernel).
No paper roofline is used for the go/no-go: we compare to the achieved same-shape GEMM."""
import subprocess
import time

import torch


def device_id():
    try:
        return subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True
        )
    except Exception as e:  # noqa: BLE001
        return f"(system_profiler unavailable: {e!r})"


def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t) / iters * 1e3  # ms


def gflops(flop, ms):
    return flop / (ms * 1e-3) / 1e9


def gemm_achieved(M, K, N, dtype=torch.float16):
    """Measured achieved GFLOP/s of a stock MPS GEMM at the conv tile's M,N,K."""
    a = torch.randn(M, K, device="mps", dtype=dtype)
    b = torch.randn(N, K, device="mps", dtype=dtype)
    ms = bench(lambda: a @ b.t())
    return ms, gflops(2 * M * K * N, ms)


def main():
    print("=== device ===")
    print(device_id())

    # conv2d 3x3, 256ch, 512x512
    x2 = torch.randn(1, 256, 512, 512, device="mps", dtype=torch.float16)
    w2 = torch.randn(256, 256, 3, 3, device="mps", dtype=torch.float16)
    ms2 = bench(lambda: torch.nn.functional.conv2d(x2, w2, padding=1))
    P2, K2, Co2 = 512 * 512, 256 * 9, 256
    gms2, gtf2 = gemm_achieved(P2, K2, Co2)
    print(f"conv2d 3x3 256ch 512x512: {ms2:.3f} ms  {gflops(2 * P2 * K2 * Co2, ms2):.1f} GFLOP/s")
    print(f"  measured GEMM @ M={P2},K={K2},N={Co2}: {gms2:.3f} ms  {gtf2:.1f} GFLOP/s "
          f"(conv2d achieves {gflops(2 * P2 * K2 * Co2, ms2) / gtf2 * 100:.0f}% of achieved GEMM)")

    # conv3d 3x3x3, 128ch, 5x256x256 (SeedVR2-ish)
    x3 = torch.randn(1, 128, 5, 256, 256, device="mps", dtype=torch.float16)
    w3 = torch.randn(128, 128, 3, 3, 3, device="mps", dtype=torch.float16)
    ms3 = bench(lambda: torch.nn.functional.conv3d(x3, w3, padding=1))
    P3, K3, Co3 = 5 * 256 * 256, 128 * 27, 128
    gms3, gtf3 = gemm_achieved(P3, K3, Co3)
    print(f"conv3d 3x3x3 128ch 5x256x256: {ms3:.3f} ms  {gflops(2 * P3 * K3 * Co3, ms3):.1f} GFLOP/s")
    print(f"  measured GEMM @ M={P3},K={K3},N={Co3}: {gms3:.3f} ms  {gtf3:.1f} GFLOP/s "
          f"(conv3d achieves {gflops(2 * P3 * K3 * Co3, ms3) / gtf3 * 100:.0f}% of achieved GEMM)")

    alloc = torch.mps.driver_allocated_memory() / 1024**3
    print(f"driver_allocated (current, NOT peak): {alloc:.2f} GiB")


if __name__ == "__main__":
    main()
