# dev/probe_matmul2d_dtype_scatter.py
"""B.0 probe (CORRECTED READOUT). The sibling probe_matmul2d_dtype.py reads the
cooperative-tensor result with a LINEAR `O[i] = c[i]`, but element i of a cooperative
tensor does NOT map to row-major position i -- it maps to (row,col) via
get_multidimensional_index. That linear readout scrambles the result and yields a
spurious ~14.5 maxdiff for EVERY dtype (uncorrelated layout), which is a readout bug,
not a kernel-capability failure. This variant scatters c[i] to O[row*N+col] exactly the
way the production gemm_nt_bias store epilogue (Section 2.3) does, and is the faithful
test of "does <T,T,float> cooperative-accumulate matmul2d run + produce correct A@B^T".
"""
import torch

_SRC = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp::tensor_ops;
kernel void probe(device @T@* A [[buffer(0)]], device @T@* B [[buffer(1)]],
                  device float* O [[buffer(2)]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr auto desc = matmul2d_descriptor(16,16,16,false,true,false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto a = tensor<device @T@, dextents<int,2>, tensor_inline>(A, dextents<int,2>{16,16}, array<int,2>{1,16});
  auto b = tensor<device @T@, dextents<int,2>, tensor_inline>(B, dextents<int,2>{16,16}, array<int,2>{1,16});
  using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(a)>;
  using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(b)>;
  auto c = op.get_destination_cooperative_tensor<AT,BT,float>();
  for (uint16_t i=0; i<c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0.0f;
  op.run(a,b,c);
  for (uint16_t i=0; i<c.get_capacity(); ++i) {
    if (!c.is_valid_element(i)) continue;
    auto idx = c.get_multidimensional_index(i);   // idx[0]=col, idx[1]=row (NT)
    const int row = int(idx[1]), col = int(idx[0]);
    O[row*16 + col] = c[i];
  }
}
"""

RESULTS = {}
for name, dt in [("half", torch.float16), ("bfloat", torch.bfloat16), ("float", torch.float32)]:
    try:
        lib = torch.mps.compile_shader(_SRC.replace("@T@", name))
        torch.manual_seed(0)
        A = torch.randn(16, 16, device="mps", dtype=dt)
        B = torch.randn(16, 16, device="mps", dtype=dt)
        O = torch.zeros(16, 16, device="mps", dtype=torch.float32)
        lib.probe(A, B, O, threads=(32, 1, 1), group_size=(32, 1, 1))
        torch.mps.synchronize()
        ref = A.float() @ B.float().t()          # NT: A @ B^T
        maxdiff = (O - ref).abs().max().item()
        nonzero = (O.abs() > 0).any().item()
        tol = {"half": 0.5, "bfloat": 1.0, "float": 1e-3}[name]
        ok = nonzero and maxdiff < tol
        RESULTS[name] = ("PASS" if ok else f"NUMFAIL maxdiff={maxdiff:.3g} nonzero={nonzero}")
    except Exception as e:
        RESULTS[name] = f"COMPILE/RUN FAIL: {e!r}"

for k, v in RESULTS.items():
    print(f"matmul2d <{k},{k},float> (scatter readout) : {v}")
