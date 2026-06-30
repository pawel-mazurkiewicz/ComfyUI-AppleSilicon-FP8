"""Issue A: Inspect what Inductor MPS actually fuses for the DiT-block tail.

Uses torch._dynamo.explain to count graphs and print break reasons.
Uses TORCH_COMPILE_DEBUG=1 to print the generated lowered IR, from which
the extern_kernels count gives the true Metal dispatch count.

Runs at BOTH production shape (2,4096,1536) and small shape (1,256,512).
Full output written to /tmp/inductor_inspect_A.txt — do not truncate with head.

Run:
  TORCH_COMPILE_DEBUG=1 \\
    /Volumes/IMPERIAL\\ SPACE/AI/ComfyUI/.venv/bin/python \\
    dev/inspect_inductor_mps_fusion.py 2>&1 | tee /tmp/inductor_inspect_A.txt
"""
import os
import sys
import torch
import torch.nn.functional as F
import torch._dynamo as dynamo

DEVICE = "mps"
DTYPE = torch.float16

SHAPES = [
    ("small (1,256,512)", 1, 256, 512),
    ("production (2,4096,1536)", 2, 4096, 1536),
]


def make(b, s, d):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    x = torch.randn(b, s, d, device=DEVICE, dtype=DTYPE, generator=g)
    weight = torch.randn(d, device=DEVICE, dtype=DTYPE, generator=g)
    W = torch.randn(d, d, device=DEVICE, dtype=DTYPE, generator=g)
    return x, weight, W


def bw_tail(x, weight, W):
    h = F.rms_norm(x, (x.shape[-1],), weight, 1e-6)
    h = F.silu(h)
    return x + h


def full_block(x, weight, W):
    h = F.rms_norm(x, (x.shape[-1],), weight, 1e-6)
    h = (h.reshape(-1, x.shape[-1]) @ W.T).reshape(x.shape)
    h = F.silu(h)
    return x + h


def report(name, fn, b, s, d):
    print(f"\n{'='*60}")
    print(f"  {name}  shape=({b},{s},{d})")
    print(f"{'='*60}")

    x, weight, W = make(b, s, d)

    # Graph structure
    torch._dynamo.reset()
    try:
        explanation = dynamo.explain(fn)(x, weight, W)
        print(f"  graph_count : {explanation.graph_count}")
        print(f"  break_reasons: {[str(r) for r in explanation.break_reasons] or 'none'}")
    except Exception as e:
        print(f"  dynamo.explain failed: {e}")

    # Lowered IR (Inductor) — printed by TORCH_COMPILE_DEBUG=1
    print(f"\n  [Compiling with Inductor "
          f"(lowered ops appear below if TORCH_COMPILE_DEBUG=1)]")
    import torch._inductor.config as ind_cfg
    old_debug = getattr(ind_cfg, "debug", False)
    try:
        ind_cfg.debug = True
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="inductor", fullgraph=False)
        compiled(x, weight, W)
        torch.mps.synchronize()
    except Exception as e:
        print(f"  compile/run failed: {e}")
    finally:
        ind_cfg.debug = old_debug


def main():
    print("Issue A — Inductor MPS fusion inspector")
    print(f"PyTorch {torch.__version__}")
    print("Full output written to /tmp/inductor_inspect_A.txt via tee.")
    print("Key things to count in Inductor debug output:")
    print("  - 'extern_kernels.' (with dot — actual method calls, e.g. extern_kernels.mm):")
    print("    each call = one separate extern MPS dispatch.")
    print("    NOTE: output_code.py always imports extern_kernels at the top — count only")
    print("    'extern_kernels.' (dot) occurrences, not the bare 'extern_kernels' import line.")
    print("  - 'compile_mps_shader': each = one fused Metal kernel (good)")
    print("  - 'ComputedBuffer' with multiple ops: fused into 1 kernel")
    print("  - rms_norm decomposition: 'aten.native_group_norm' or manual ops?")
    print()

    for shape_name, b, s, d in SHAPES:
        for fn_name, fn in [("bw_tail", bw_tail), ("full_block", full_block)]:
            report(f"{fn_name}", fn, b, s, d)

    print("\nDone. Paste extern_kernels. (dot) call count for each shape into docs/superpowers/results/A-results.md.")
    print("PASS threshold: bw_tail production shape extern_kernels. actual calls == 0 (all fused into Metal shader).")
    print("NOTE: output_code.py always has 'from ... import extern_kernels' — that import line does NOT count.")


if __name__ == "__main__":
    main()
