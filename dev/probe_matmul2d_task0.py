#!/usr/bin/env python3
"""G2 Task 0 — empirical compile-only probes for matmul2d dtype facts.

Compiles tiny single-kernel Metal sources under MTLLanguageVersion4_1 and records,
per fact, whether the relevant type / instantiation / macro is available on THIS SDK.
Gates the downstream per-dtype dispatch probe (dev/probe_matmul2d_dtypes.py).
"""
from __future__ import annotations
import sys
# Reuse the extension builder + source templates from the main probe module.
import importlib.util, pathlib
_p = pathlib.Path(__file__).parent / "probe_matmul2d_dtypes.py"
_spec = importlib.util.spec_from_file_location("probe_matmul2d_dtypes", _p)
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)


def _hdr() -> str:
    return (
        "#include <metal_stdlib>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "using namespace metal;\n"
        "using namespace mpp::tensor_ops;\n"
    )


def main():
    import torch
    if not torch.backends.mps.is_available():
        print("ERROR: MPS not available.", file=sys.stderr); sys.exit(1)
    mod = _probe.build_extension()
    if mod is None:
        print("ERROR: host extension build failed; see stderr.", file=sys.stderr); sys.exit(1)

    probes = []

    # 0.3a — does <half,half,float> get_destination_cooperative_tensor compile?
    for ty in ("half", "bfloat", "float"):
        src = _probe._metal_src(ty)              # full single-kernel source for this dtype
        r = mod.compile_probe(src, "probe_kernel")
        probes.append((f"matmul2d<{ty}> + <{ty},{ty},float> accum compiles",
                       bool(r["compile_ok"] and r["has_fn"]), r["compile_error"]))

    # 0.3b — signed char control compiles
    r = mod.compile_probe(_probe._metal_src("signed_char"), "probe_kernel")
    probes.append(("matmul2d<signed char> + int accum compiles",
                   bool(r["compile_ok"] and r["has_fn"]), r["compile_error"]))

    # 0.4 — fp8 e4m3 / e5m2 SEPARATE macro availability (verbatim from Codex review item 8).
    # We compile the type-reference under each macro and look up the function: if the macro
    # is undefined the kernel body is empty -> compiles but has_fn detection below.
    fp8_probe = lambda macro, ty: _hdr() + (
        "kernel void p(device uchar* x [[buffer(0)]]) {\n"
        f"#if {macro}\n"
        f"  using t = metal::{ty};\n"
        "  device t* y = (device t*)x;\n"
        "  (void)y;\n"
        "#else\n"
        "  // macro absent on this SDK; reference a compile-time failure marker so the\n"
        "  // probe can DISTINGUISH 'macro absent' from 'type present but broken'.\n"
        "  x[0] = x[0];\n"
        "#endif\n"
        "}\n"
    )
    for macro, ty, label in (
        ("__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__", "metal_fp8_e4m3_format", "fp8 e4m3 type"),
        ("__HAVE_METAL_FP8_E5M2_FORMAT_TYPE__", "metal_fp8_e5m2_format", "fp8 e5m2 type"),
    ):
        # Macro-true variant: only compiles cleanly if both macro defined AND type valid.
        src_true = _hdr() + (
            "kernel void p(device uchar* x [[buffer(0)]]) {\n"
            f"  using t = metal::{ty};\n"
            "  device t* y = (device t*)x; (void)y;\n"
            "}\n"
        )
        r = mod.compile_probe(src_true, "p")
        probes.append((f"{label} available under Metal 4.1",
                       bool(r["compile_ok"] and r["has_fn"]), r["compile_error"]))

    print("\n=== G2 Task 0 — compile-only probe results ===")
    lines = ["", "## S-G2-Task0 — matmul2d dtype API facts (compile-only)", "",
             "| fact | result | error |", "|------|--------|-------|"]
    for label, ok, err in probes:
        verdict = "PASS" if ok else "FAIL"
        e = (err or "").replace("\n", " ")[:80] or "—"
        print(f"  {label:<52} {verdict}   {e}")
        lines.append(f"| {label} | {verdict} | {e} |")
    section = "\n".join(lines)

    facts = pathlib.Path(__file__).parent.parent / "INVESTIGATION_FACTS.md"
    if facts.exists():
        with open(facts, "a") as f:
            f.write("\n" + section + "\n")
        print(f"\n[task0] appended to {facts}")
    else:
        print(f"\n[task0] WARNING: {facts} not found; not appended.", file=sys.stderr)


if __name__ == "__main__":
    main()
