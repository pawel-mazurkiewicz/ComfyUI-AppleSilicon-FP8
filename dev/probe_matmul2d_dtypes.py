#!/usr/bin/env python3
"""G2 — M5 matmul2d operand dtype capability probe.

For each candidate dtype (signed char/half/bfloat/float/fp8_e4m3/fp8_e5m2) this
generates a SINGLE-KERNEL Metal source, compiles it IN ISOLATION under Metal 4.1
(so one dtype's compile failure never aborts the others), dispatches a tiny matmul,
and records compile/run/correctness vs a torch fp32 reference computed from the
DECODED stored operands. Appends the capability matrix to INVESTIGATION_FACTS.md.

Usage:
    python dev/probe_matmul2d_dtypes.py            # run + append to INVESTIGATION_FACTS.md
    python dev/probe_matmul2d_dtypes.py --dry-run  # print only, do not write

Gates: issue B (conv GEMM precision). Does NOT gate W4A8 (sub-byte int4 — separate probe).
Autonomously runnable on MPS — no ComfyUI, no model weights.
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Generic ObjC++ host: compile_probe(src, fn) and probe_one(src, fn, A,B,C,...).
# NO per-dtype kernels embedded here — Python generates one source per dtype and
# passes it in, so each dtype compiles in its own MTLLibrary (Codex BLOCKER 1),
# and Metal compile success is reported via has_fn from newFunctionWithName
# (Codex BLOCKER 2), never via Python binding presence.
# ---------------------------------------------------------------------------

_OBJCPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
using namespace at::mps;

static id<MTLLibrary> compile_lib(const std::string& src, std::string& err_out) {
    id<MTLDevice> dev = MPSDevice::getInstance()->device();
    MTLCompileOptions* opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion4_1;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        [NSString stringWithUTF8String:src.c_str()] options:opts error:&err];
    if (!lib) {
        err_out = std::string(err && err.localizedDescription
            ? err.localizedDescription.UTF8String : "unknown compile error");
    }
    return lib;
}

// Compile-only probe (Task 0). Returns compile_ok / compile_error / has_fn.
static py::dict compile_probe(const std::string& src, const std::string& fn_name) {
    py::dict r;
    r["compile_ok"] = false; r["compile_error"] = std::string("");
    r["has_fn"] = false;
    std::string cerr;
    id<MTLLibrary> lib = compile_lib(src, cerr);
    if (!lib) { r["compile_error"] = cerr; return r; }
    r["compile_ok"] = true;
    if (!fn_name.empty()) {
        id<MTLFunction> fn = [lib newFunctionWithName:
            [NSString stringWithUTF8String:fn_name.c_str()]];
        r["has_fn"] = (fn != nil);
    }
    return r;
}

// Compile-in-isolation + dispatch. Returns per-stage status so a failure in any
// stage still yields a structured row instead of aborting the whole matrix.
static py::dict probe_one(const std::string& src, const std::string& fn_name,
                          torch::Tensor A, torch::Tensor B, torch::Tensor C,
                          int64_t M, int64_t N, int64_t K, int64_t NSG) {
    py::dict r;
    r["compile_ok"] = false; r["compile_error"] = std::string("");
    r["has_fn"] = false;
    r["pso_ok"] = false; r["pso_error"] = std::string("");
    r["run_ok"] = false; r["run_error"] = std::string("");

    std::string cerr;
    id<MTLLibrary> lib = compile_lib(src, cerr);
    if (!lib) { r["compile_error"] = cerr; return r; }
    r["compile_ok"] = true;

    NSString* ns = [NSString stringWithUTF8String:fn_name.c_str()];
    id<MTLFunction> fn = [lib newFunctionWithName:ns];
    if (!fn) { return r; }   // compiled but kernel guarded out -> has_fn=false (COMPILE_SKIP)
    r["has_fn"] = true;

    NSError* perr = nil;
    id<MTLDevice> dev = MPSDevice::getInstance()->device();
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&perr];
    if (!pso) {
        r["pso_error"] = std::string(perr && perr.localizedDescription
            ? perr.localizedDescription.UTF8String : "unknown pso error");
        return r;
    }
    r["pso_ok"] = true;

    // ---- validation BEFORE any bit-cast (Codex BLOCKER 10) ----
    TORCH_CHECK(A.is_mps() && B.is_mps() && C.is_mps(), "A,B,C must be MPS tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
                "A,B,C must be contiguous");
    TORCH_CHECK(A.dim()==2 && B.dim()==2 && C.dim()==2, "A,B,C must be 2-D");
    TORCH_CHECK(A.size(0)==M && A.size(1)==K, "A must be [M,K]");
    TORCH_CHECK(B.size(0)==N && B.size(1)==K, "B must be [N,K]");
    TORCH_CHECK(C.size(0)==M && C.size(1)==N, "C must be [M,N]");

    MPSStream* stream = getCurrentMPSStream();
    id<MTLBuffer> aBuf = __builtin_bit_cast(id<MTLBuffer>, A.storage().data());
    id<MTLBuffer> bBuf = __builtin_bit_cast(id<MTLBuffer>, B.storage().data());
    id<MTLBuffer> cBuf = __builtin_bit_cast(id<MTLBuffer>, C.storage().data());
    const NSUInteger aOff = A.storage_offset()*A.element_size();
    const NSUInteger bOff = B.storage_offset()*B.element_size();
    const NSUInteger cOff = C.storage_offset()*C.element_size();
    int Mi=(int)M, Ni=(int)N, Ki=(int)K;
    const NSUInteger gx=(NSUInteger)((M+63)/64), gy=(NSUInteger)((N+63)/64);
    @try {
        dispatch_sync(stream->queue(), ^(){
            @autoreleasepool {
                id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
                [enc setComputePipelineState:pso];
                [enc setBuffer:aBuf offset:aOff atIndex:0];
                [enc setBuffer:bBuf offset:bOff atIndex:1];
                [enc setBuffer:cBuf offset:cOff atIndex:2];
                [enc setBytes:&Mi length:sizeof(int) atIndex:3];
                [enc setBytes:&Ni length:sizeof(int) atIndex:4];
                [enc setBytes:&Ki length:sizeof(int) atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(gx,gy,1)
                    threadsPerThreadgroup:MTLSizeMake((NSUInteger)(NSG*32),1,1)];
            }
        });
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
        r["run_ok"] = true;
    } @catch (NSException* ex) {
        r["run_error"] = std::string(ex.reason ? ex.reason.UTF8String : "objc exception");
    }
    return r;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compile_probe", &compile_probe, "compile one Metal source; report compile_ok/has_fn");
    m.def("probe_one", &probe_one, "compile one source in isolation and dispatch it");
}
"""


# ---------------------------------------------------------------------------
# Per-dtype Metal source generation (one single-kernel source per candidate).
# ---------------------------------------------------------------------------

_HDR = (
    "#include <metal_stdlib>\n"
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    "using namespace metal;\n"
    "using namespace mpp::tensor_ops;\n"
    "// Cooperative tensors cannot cross a function boundary; write-out via macro.\n"
    "#define WRITEOUT(Cb, cC, m0, n0, M, N) \\\n"
    "    for (uint16_t _i=0;_i<cC.get_capacity();++_i) { \\\n"
    "        if (!cC.is_valid_element(_i)) continue; \\\n"
    "        auto _idx = cC.get_multidimensional_index(_i); \\\n"
    "        const int _r=int(_idx[1]),_c=int(_idx[0]); \\\n"
    "        if (m0+_r>=M||n0+_c>=N) continue; \\\n"
    "        Cb[ulong(_r)*N+_c]=cC[_i]; }\n"
)

# (metal element type, C buffer type, accumulator type, zero-literal)
_TYPED = {
    "signed_char": ("signed char", "int",   "int",   "0"),
    "half":        ("half",        "float", "float", "0.0f"),
    "bfloat":      ("bfloat",      "float", "float", "0.0f"),
    "float":       ("float",       "float", "float", "0.0f"),
}

# fp8 candidate -> (metal type, macro)
_FP8 = {
    "fp8_e4m3": ("metal_fp8_e4m3_format", "__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__"),
    "fp8_e5m2": ("metal_fp8_e5m2_format", "__HAVE_METAL_FP8_E5M2_FORMAT_TYPE__"),
}


def _metal_src(dtype_name: str) -> str:
    """Return a complete single-kernel Metal source named `probe_kernel` for one dtype."""
    if dtype_name in _TYPED:
        elem, cty, accum, zero = _TYPED[dtype_name]
        body = f"""
kernel void probe_kernel(
    device {elem}* A [[buffer(0)]], device {elem}* B [[buffer(1)]],
    device {cty}* C [[buffer(2)]],
    constant int& M [[buffer(3)]], constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]], uint3 tgid [[threadgroup_position_in_grid]])
{{
    constexpr int BM=64, BN=64, NSG=4;
    const int m0=int(tgid.x)*BM, n0=int(tgid.y)*BN;
    if (m0>=M||n0>=N) return;
    constexpr auto desc = matmul2d_descriptor(BM,BN,static_cast<int>(dynamic_extent),
        false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<NSG>> op;
    auto mA=tensor<device {elem},dextents<int,2>,tensor_inline>(
                A+ulong(m0)*K,dextents<int,2>{{K,min(BM,M-m0)}},array<int,2>{{1,K}});
    auto mB=tensor<device {elem},dextents<int,2>,tensor_inline>(
                B+ulong(n0)*K,dextents<int,2>{{K,min(BN,N-n0)}},array<int,2>{{1,K}});
    using AT=__tensor_ops_detail::__remove_addrspace_t<decltype(mA)>;
    using BT=__tensor_ops_detail::__remove_addrspace_t<decltype(mB)>;
    auto cC=op.get_destination_cooperative_tensor<AT,BT,{accum}>();
    for (uint16_t i=0;i<cC.get_capacity();++i) if(cC.is_valid_element(i)) cC[i]={zero};
    op.run(mA,mB,cC);
    device {cty}* Cb=C+ulong(m0)*N+n0;
    WRITEOUT(Cb,cC,m0,n0,M,N)
}}
"""
        return _HDR + body

    if dtype_name in _FP8:
        mtype, macro = _FP8[dtype_name]
        body = f"""
#if defined({macro})
kernel void probe_kernel(
    device uchar* rawA [[buffer(0)]], device uchar* rawB [[buffer(1)]],
    device float* C    [[buffer(2)]],
    constant int& M [[buffer(3)]], constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]], uint3 tgid [[threadgroup_position_in_grid]])
{{
    using fp8_t = metal::{mtype};
    constexpr int BM=64, BN=64, NSG=4;
    const int m0=int(tgid.x)*BM, n0=int(tgid.y)*BN;
    if (m0>=M||n0>=N) return;
    constexpr auto desc = matmul2d_descriptor(BM,BN,static_cast<int>(dynamic_extent),
        false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<NSG>> op;
    // fp8 tensor data_handle_type is `device uchar*` (1 byte/elem), so pass the
    // raw uchar* directly — NOT cast to fp8_t* (mirrors production gemm_fp8_nt,
    // _patches/fp8_ext/fp8_matmul2d.mm:96).
    auto mA=tensor<device fp8_t,dextents<int,2>,tensor_inline>(
                rawA+ulong(m0)*K,
                dextents<int,2>{{K,min(BM,M-m0)}},array<int,2>{{1,K}});
    auto mB=tensor<device fp8_t,dextents<int,2>,tensor_inline>(
                rawB+ulong(n0)*K,
                dextents<int,2>{{K,min(BN,N-n0)}},array<int,2>{{1,K}});
    using AT=__tensor_ops_detail::__remove_addrspace_t<decltype(mA)>;
    using BT=__tensor_ops_detail::__remove_addrspace_t<decltype(mB)>;
    auto cC=op.get_destination_cooperative_tensor<AT,BT,float>();
    for (uint16_t i=0;i<cC.get_capacity();++i) if(cC.is_valid_element(i)) cC[i]=0.0f;
    op.run(mA,mB,cC);
    device float* Cb=C+ulong(m0)*N+n0;
    WRITEOUT(Cb,cC,m0,n0,M,N)
}}
#endif  // {macro}
"""
        return _HDR + body

    raise ValueError(f"unknown dtype {dtype_name!r}")


# ---------------------------------------------------------------------------
# Extension builder (mirrors _patches/int8_ext/loader.py)
# ---------------------------------------------------------------------------

_NOSPACE_ROOT = "/tmp/asfp8_build"
_BUILD_DIR    = os.path.join(_NOSPACE_ROOT, "probe_ext")


def _nospace_torch_lib() -> str:
    """No-space symlink to torch/lib (fixes cpp_extension unquoted -L on this path)."""
    import torch.utils.cpp_extension as cpp
    real = cpp.TORCH_LIB_PATH
    if " " not in real:
        return real
    os.makedirs(_NOSPACE_ROOT, exist_ok=True)
    link = os.path.join(_NOSPACE_ROOT, "torchlib")
    if os.path.islink(link):
        os.unlink(link)
    os.symlink(real, link)
    return link


def _ensure_ninja() -> bool:
    if shutil.which("ninja"):
        return True
    try:
        import ninja
        bin_dir = getattr(ninja, "BIN_DIR", None) or os.path.join(
            os.path.dirname(ninja.__file__), "data", "bin"
        )
        if os.path.isdir(bin_dir):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:
        print(f"[probe_dtypes] ninja import failed: {e!r}", file=sys.stderr)
    return bool(shutil.which("ninja"))


def build_extension():
    """Write the generic ObjC++ host to a no-space temp path and compile via cpp_extension."""
    if shutil.which("xcrun") is None:
        print("[probe_dtypes] xcrun not found — install Xcode CLI tools.", file=sys.stderr)
        return None
    if not _ensure_ninja():
        print("[probe_dtypes] ninja binary not on PATH (pip install ninja).", file=sys.stderr)
        return None
    try:
        import torch.utils.cpp_extension as cpp
        from torch.utils.cpp_extension import load as cpp_load
    except Exception as e:
        print(f"[probe_dtypes] cpp_extension unavailable: {e!r}", file=sys.stderr)
        return None

    os.makedirs(_BUILD_DIR, exist_ok=True)
    src_path = os.path.join(_BUILD_DIR, "probe_matmul2d_host.mm")
    Path(src_path).write_text(_OBJCPP_SRC)

    saved = cpp.TORCH_LIB_PATH
    try:
        cpp.TORCH_LIB_PATH = _nospace_torch_lib()
        mod = cpp_load(
            name="asfp8_probe_host",
            sources=[src_path],
            extra_cflags=["-std=c++17", "-ObjC++"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            build_directory=_BUILD_DIR,
            verbose=False,
        )
    except Exception as e:
        print(f"[probe_dtypes] build FAILED: {e!r}", file=sys.stderr)
        mod = None
    finally:
        cpp.TORCH_LIB_PATH = saved
    return mod


# ---------------------------------------------------------------------------
# Per-dtype probe helpers
# ---------------------------------------------------------------------------

def _make_fp8_uint8(shape, dtype_str):
    """uint8 tensor on MPS whose bytes are fp8 values, plus the decoded fp32 reference.

    CPU torch float8 cast (MPS cannot cast fp8), then byte-view to uint8 -> MPS.
    The decoded reference is the SAME quantized tensor cast back to float, matching
    the principle of _patches/_common.decode_fp8. No clamping.
    """
    import torch
    fp32 = torch.randn(shape) * 0.5
    fp8_dtype = torch.float8_e4m3fn if "e4m3" in dtype_str else torch.float8_e5m2
    fp8_cpu = fp32.to(fp8_dtype)
    return fp8_cpu.view(torch.uint8).to("mps"), fp8_cpu.float()


def run_probe(mod, dtype_name, M=64, N=64, K=128):
    """Run one dtype probe. Returns dict:
    compile_ok, has_fn, pso_ok, run_ok, max_err, exact, latency_ms, verdict, detail.
    verdict in {PASS, FAIL, COMPILE_FAIL, COMPILE_SKIP, RUN_FAIL}.
    """
    import torch
    res = dict(compile_ok=False, has_fn=False, pso_ok=False, run_ok=False,
               max_err=float("nan"), exact=False, latency_ms=float("nan"),
               verdict="", detail="")

    cfg = {
        "signed_char": (torch.int8,    torch.int32),
        "half":        (torch.float16,  torch.float32),
        "bfloat":      (torch.bfloat16, torch.float32),
        "float":       (torch.float32,  torch.float32),
        "fp8_e4m3":    ("fp8",          torch.float32),
        "fp8_e5m2":    ("fp8",          torch.float32),
    }
    if dtype_name not in cfg:
        res["verdict"] = "FAIL"; res["detail"] = "unknown dtype"; return res
    op_dtype, c_dtype = cfg[dtype_name]

    src = _metal_src(dtype_name)

    # -- build operands (reference uses DECODED stored values for EVERY dtype) ----
    torch.manual_seed(42)
    if dtype_name.startswith("fp8"):
        A_mps, A_ref = _make_fp8_uint8((M, K), dtype_name)
        B_mps, B_ref = _make_fp8_uint8((N, K), dtype_name)
        A_mps = A_mps.contiguous(); B_mps = B_mps.contiguous()
    elif dtype_name == "signed_char":
        A_cpu = torch.randint(-64, 64, (M, K), dtype=torch.int8)
        B_cpu = torch.randint(-64, 64, (N, K), dtype=torch.int8)
        A_mps = A_cpu.to("mps").contiguous(); B_mps = B_cpu.to("mps").contiguous()
        A_ref = A_cpu.float(); B_ref = B_cpu.float()
    else:
        A_mps = torch.randn(M, K).to(op_dtype).to("mps").contiguous()
        B_mps = torch.randn(N, K).to(op_dtype).to("mps").contiguous()
        # Codex MAJOR 4: reference = decoded STORED operands, not unrounded fp32.
        A_ref = A_mps.cpu().float(); B_ref = B_mps.cpu().float()

    C_mps = torch.zeros(M, N, dtype=c_dtype, device="mps").contiguous()
    ref = A_ref @ B_ref.t()

    # -- compile-in-isolation + dispatch ----------------------------------------
    t0 = time.perf_counter()
    st = mod.probe_one(src, "probe_kernel", A_mps, B_mps, C_mps, M, N, K, 4)
    torch.mps.synchronize()
    res["latency_ms"] = (time.perf_counter() - t0) * 1e3
    res["compile_ok"] = bool(st["compile_ok"])
    res["has_fn"]     = bool(st["has_fn"])
    res["pso_ok"]     = bool(st["pso_ok"])
    res["run_ok"]     = bool(st["run_ok"])

    if not res["compile_ok"]:
        res["verdict"] = "COMPILE_FAIL"; res["detail"] = st["compile_error"]; return res
    if not res["has_fn"]:
        # compiled but kernel guarded out (fp8 macro absent, etc.)
        res["verdict"] = "COMPILE_SKIP"
        res["detail"] = "kernel absent (macro/type not available on this SDK)"
        return res
    if not res["pso_ok"]:
        res["verdict"] = "COMPILE_FAIL"; res["detail"] = st["pso_error"]; return res
    if not res["run_ok"]:
        res["verdict"] = "RUN_FAIL"; res["detail"] = st["run_error"]; return res

    # -- correctness (+ SPY: prove the real kernel ran, not a no-op) ------------
    out = C_mps.cpu()
    if dtype_name == "signed_char":
        ref_i32 = ref.round().to(torch.int32)
        res["exact"] = bool(torch.equal(out, ref_i32))
        res["max_err"] = float((out.float() - ref).abs().max().item())
        # SPY: a no-op/fallback would leave C all-zero; a real GEMM does not.
        nonzero = bool(out.abs().sum().item() != 0)
        if res["exact"] and nonzero:
            res["verdict"] = "PASS"
        else:
            res["verdict"] = "FAIL"
            res["detail"] = (f"exact={res['exact']} nonzero={nonzero} "
                             f"max_err={res['max_err']:.3g}")
        return res

    out_f = out.float()
    res["max_err"] = float((out_f - ref).abs().max().item())
    tol = {"half": 0.5, "bfloat": 1.0, "float": 1e-3,
           "fp8_e4m3": 0.5, "fp8_e5m2": 0.5}[dtype_name]
    # SPY: C must differ from the all-zero init for a real kernel run.
    nonzero = bool(out_f.abs().sum().item() != 0)
    if res["max_err"] <= tol and nonzero:
        res["verdict"] = "PASS"
    else:
        res["verdict"] = "FAIL"
        res["detail"] = f"max_err={res['max_err']:.3g} > tol={tol} or nonzero={nonzero}"
    return res


# ---------------------------------------------------------------------------
# Gate decisions (Codex MAJOR 6: fp8 PASS gates an FP8 path ONLY, not W4A8).
# ---------------------------------------------------------------------------

CANDIDATES = ["signed_char", "half", "bfloat", "float", "fp8_e4m3", "fp8_e5m2"]

DECISIONS = {
    "half":     "issue B conv GEMM can use fp16 operands on tensor units",
    "bfloat":   "issue B conv GEMM can use bf16 operands on tensor units",
    "float":    "fp32 cooperative path legal (perf NOT proven here)",
    "fp8_e4m3": "FP8 e4m3 operand path viable (regresses existing fp8fp8_matmul2d_nt); NOT W4A8",
    "fp8_e5m2": "FP8 e5m2 operand path viable (informational); NOT W4A8",
}


def build_section(rows):
    lines = [
        "",
        "## S-G2 — M5 matmul2d operand dtype support (probe_matmul2d_dtypes.py)",
        "",
        "Machine: M5 Max / macOS 27 / PyTorch 2.11 / Metal 4.1 (MTLLanguageVersion4_1).",
        "NT layout (A[M,K] @ Bᵀ, B[N,K]), BM=BN=64, NSG=4, K=128.",
        "Reference: torch fp32 matmul of DECODED stored operands. Each dtype compiled in",
        "its own MTLLibrary (a compile failure of one dtype does not affect the others).",
        "Latency is INFORMATIONAL ONLY (launch-dominated; not a tensor-unit perf signal).",
        "",
        "| dtype        | compile | run  | max_err | latency | verdict | gates (issue B / FP8 only) |",
        "|--------------|---------|------|---------|---------|---------|----------------------------|",
    ]
    for r in rows:
        d = r["name"]
        c = "OK" if r["res"]["compile_ok"] else "FAIL"
        run = "OK" if r["res"]["run_ok"] else ("—" if not r["res"]["has_fn"] else "FAIL")
        e = r["res"]["max_err"]
        e_s = f"{e:.3g}" if e == e else "—"
        lat = r["res"]["latency_ms"]
        l_s = f"{lat:.2f}ms" if lat == lat else "—"
        verdict = r["res"]["verdict"] or "—"
        gate = DECISIONS.get(d, "—")
        lines.append(f"| {d:<12} | {c:<7} | {run:<4} | {e_s:<7} | {l_s:<7} | {verdict:<7} | {gate} |")
    lines += [
        "",
        "**Decision rule:**",
        "- half or bfloat PASS → issue B conv GEMM can use that dtype on tensor units.",
        "- half AND bfloat both FAIL → issue B must keep int8 operands (current path).",
        "- fp8_e4m3 PASS → confirms the existing fp8×fp8 NT path (fp8fp8_matmul2d_nt) regression.",
        "- fp8_e5m2 PASS → e5m2 operand path is usable (informational).",
        "- W4A8 is NOT decided here: it needs a separate int4/uint4 packing + correctness probe.",
        "- float PASS proves legality ONLY; it does NOT prove tensor-unit performance.",
        "",
        f"Probe run: {time.strftime('%Y-%m-%d %H:%M %Z')}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="G2 matmul2d dtype probe")
    parser.add_argument("--dry-run", action="store_true",
                        help="print results without appending to INVESTIGATION_FACTS.md")
    args = parser.parse_args()

    import torch
    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available on this machine.", file=sys.stderr)
        sys.exit(1)

    print("[probe_dtypes] building host extension (first run: ~30 s for JIT compile) …")
    mod = build_extension()
    if mod is None:
        print("ERROR: extension build failed; see stderr above.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for d in CANDIDATES:
        print(f"  probing {d} …", end=" ", flush=True)
        r = run_probe(mod, d)
        e = f"{r['max_err']:.3g}" if r["max_err"] == r["max_err"] else "—"
        lat = f"{r['latency_ms']:.2f}ms" if r["latency_ms"] == r["latency_ms"] else "—"
        print(f"compile={'T' if r['compile_ok'] else 'F'} has_fn={'T' if r['has_fn'] else 'F'} "
              f"run={'T' if r['run_ok'] else 'F'}  max_err={e}  lat={lat}  {r['verdict']} "
              f"{('('+r['detail']+')') if r['detail'] else ''}")
        rows.append({"name": d, "res": r})

    section = build_section(rows)
    print()
    print(section)

    if not args.dry_run:
        facts_path = Path(__file__).parent.parent / "INVESTIGATION_FACTS.md"
        if facts_path.exists():
            with open(facts_path, "a") as f:
                f.write("\n" + section + "\n")
            print(f"[probe_dtypes] appended to {facts_path}")
        else:
            print(f"[probe_dtypes] WARNING: {facts_path} not found; did not append.",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
