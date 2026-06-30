# dev/probe_compile_shader_dispatch.py
"""B.0b probe: (1) device-int* PRM scalars arrive intact under compile_shader,
(2) the 2D threadgroup grid maps with no axis transposition."""
import torch

_PRM_SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void prm_echo(device const int* PRM [[buffer(0)]], device int* O [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < 5) O[gid] = PRM[gid];
}
"""
lib = torch.mps.compile_shader(_PRM_SRC)
prm = torch.tensor([7, 11, 13, 17, 19], dtype=torch.int32, device="mps")
out = torch.zeros(5, dtype=torch.int32, device="mps")
lib.prm_echo(prm, out, threads=(32, 1, 1), group_size=(32, 1, 1))
torch.mps.synchronize()
print("PRM echo:", out.cpu().tolist(), "expect [7, 11, 13, 17, 19] ->",
      "PASS" if out.cpu().tolist() == [7, 11, 13, 17, 19] else "FAIL")

_GRID_SRC = """
#include <metal_stdlib>
using namespace metal;
// write the (tx,ty) threadgroup coords of each block into G[ty*GX + tx]
kernel void grid_echo(device int* G [[buffer(0)]], device const int* DIM [[buffer(1)]],
                      uint3 tg [[threadgroup_position_in_grid]]) {
    const int GX = DIM[0];
    G[int(tg.y) * GX + int(tg.x)] = int(tg.x) * 100 + int(tg.y);
}
"""
gx, gy, TG = 3, 2, 32
lib2 = torch.mps.compile_shader(_GRID_SRC)
G = torch.full((gx * gy,), -1, dtype=torch.int32, device="mps")
DIM = torch.tensor([gx, gy], dtype=torch.int32, device="mps")
lib2.grid_echo(G, DIM, threads=(gx * TG, gy, 1), group_size=(TG, 1, 1))
torch.mps.synchronize()
got = G.cpu().tolist()
want = [tx * 100 + ty for ty in range(gy) for tx in range(gx)]
print("grid echo:", got, "expect", want, "->", "PASS" if got == want else "FAIL (axis transposed?)")
