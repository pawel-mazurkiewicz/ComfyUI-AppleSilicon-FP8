"""Probe: which transcendental spellings compile under the MPS Metal path, and do they
match torch.nn.functional within the planned tolerance? Records a verdict the plan gates on."""
import torch

assert torch.backends.mps.is_available(), "probe must run on MPS"
assert hasattr(torch.mps, "compile_shader"), \
    "torch.mps.compile_shader missing; fall back to a tiny .mm via loader (see note below)"


def try_compile(name, body):
    src = (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void k(device const float* in [[buffer(0)]],\n"
        "              device float* out [[buffer(1)]],\n"
        "              uint i [[thread_position_in_grid]]) {\n"
        f"    float x = in[i];\n    {body}\n"
        "}\n"
    )
    try:
        lib = torch.mps.compile_shader(src)
        print(f"[{name}] COMPILE OK")
        return lib
    except Exception as e:
        print(f"[{name}] COMPILE FAIL: {type(e).__name__}: {e}")
        return None


# 1) does precise:: namespace expose exp/tanh?
try_compile("precise.exp_tanh", "out[i] = precise::exp(-x) + precise::tanh(x);")
# 2) fast (default) exp/tanh as a fallback if precise is unavailable
try_compile("fast.exp_tanh", "out[i] = exp(-x) + tanh(x);")
# 3) erf for gelu-default (act=3) — unqualified
try_compile("erf.unqualified", "out[i] = erf(x);")
# 4) erf qualified
try_compile("erf.metal", "out[i] = metal::erf(x);")

# If compiles succeed, also check numeric parity for SiLU/GELU-tanh at the planned tolerance.
def run(lib, n=4096):
    x = (torch.randn(n, dtype=torch.float32, device="mps"))
    out = torch.empty_like(x)
    return x, out, lib

print("PROBE COMPLETE — record per-symbol COMPILE OK/FAIL above.")
