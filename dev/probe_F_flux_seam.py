"""Issue-F Task -1 — HARD PRE-WIRE GATE (HUMAN-REQUIRED): probe a real Flux-2-Klein load.

This script CANNOT be run autonomously by the implementing agent — it needs Flux-2-Klein-9B
(fp8) ACTUALLY LOADED in ComfyUI. The repo's INVESTIGATION_FACTS.md (S16) records that a prior
fp8-Linear seam assumption was WRONG, so patch #20 (fp8_linear_kernel_mps) ships DEFAULT OFF and
its wiring is treated as UNVALIDATED until this probe confirms three things on a live model:

  (1) THE SEAM  — which class actually owns the fp8 Linear.forward, and that it is entered
                  BEFORE torch._scaled_mm (i.e. before the activation is fp8-quantized).
  (2) SCALE SHAPE — the weight dequant scale is scalar (numel 1) vs per-output-channel [N].
  (3) ACT RANGE — real per-layer |x|.max() vs fp16's 65504 (the wrapper casts bf16->half).

HOW TO RUN (human):
  1. Launch ComfyUI with the env:
        ASFP8_FP8_NATIVE=1 ASFP8_PROFILE=1 <comfyui launch>
  2. Load a Flux-2-Klein-9B fp8 workflow and queue ONE 1024^2 render.
  3. Either: paste the body below into a ComfyUI python console after the model object is in
     scope, OR import this module and call probe(model) from a custom node / breakpoint, where
     `model` is the DiT module (e.g. the loaded UNet/transformer torch.nn.Module).

The probe is READ-ONLY: it installs temporary hooks/wrappers, runs/observes ONE step, then
restores everything. It prints four findings and (if pointed at docs/) can be pasted into
docs/superpowers/results/F-results.md under "Task -1 (real-model gate)".

DECISION MATRIX (drives whether patch #20 can be enabled):
  seam class   : MRO includes comfy.ops...mixed_precision_ops...Linear AND FORWARD logs BEFORE
                 scaled_mm  -> proceed (wrapper target as written).
                 If fp8_ops.Linear -> re-point install() at fp8_ops.Linear.forward_comfy_cast_weights.
                 If manual_cast.Linear -> OUT OF SCOPE, stop.
                 If FORWARD never logs before scaled_mm -> activation quantized upstream; seam wrong, stop.
  scale shape  : () or (1,) -> scalar branch.  (N,) -> per-channel branch. Both already coded.
  layout_cls   : TensorCoreFP8Layout / ...E4M3Layout + _qdata.dtype==float8_e4m3fn -> eligible.
                 E5M2 / float8_e5m2 -> NOT eligible (kernel is e4m3-only); those layers fall back.
  act |max|    : all << 65504 -> leave range guard off, document the bound.
                 any >~ 65504 -> enable ASFP8_FP8_NATIVE_RANGE_GUARD=1, record outlier layers.
"""
import torch


def probe(model, run_one_step=None):
    """Run the three-part probe against a live Flux-2 `model` (an nn.Module).

    Parameters
    ----------
    model : torch.nn.Module
        The loaded DiT (transformer) whose fp8 Linear layers we want to inspect.
    run_one_step : callable | None
        Optional zero-arg callable that triggers exactly ONE forward step of the model
        (e.g. a single sampler step). If None, the OPTRACE and ACT-RANGE sections that need
        a live forward are skipped with a warning; the static SEAM+SCALE dump still runs.
    """
    # ---------------------------------------------------------------------------------------
    # (1) SEAM + SCALE SHAPE: find one eligible DiT fp8 Linear and dump its structure.
    # ---------------------------------------------------------------------------------------
    target = None
    for name, m in model.named_modules():
        w = getattr(m, "weight", None)
        if hasattr(w, "_layout_cls") and str(w._layout_cls).startswith("TensorCoreFP8"):
            print("NAME      :", name)
            print("MRO       :", type(m).__mro__)            # <-- IS mixed_precision_ops.Linear in here?
            print("layout_cls:", w._layout_cls)              # TensorCoreFP8Layout / ...E4M3Layout / ...E5M2Layout
            print("qdata     :", tuple(w._qdata.shape), w._qdata.dtype, w._qdata.device)
            sc = w._params.scale
            sc_shape = tuple(sc.shape) if hasattr(sc, "shape") else "scalar"
            print("scale     :", sc_shape, "numel", sc.numel(), sc.dtype, sc.device)
            print("quant_fmt :", getattr(m, "quant_format", None), "layout_type:", getattr(m, "layout_type", None))
            target = m
            break
    if target is None:
        print("PROBE: no fp8 (TensorCoreFP8*) Linear found in this model — wrong model or wrong seam.")
        return

    if run_one_step is None:
        print("PROBE: run_one_step not provided; skipping OPTRACE + ACT-RANGE (need a live forward).")
        return

    # ---------------------------------------------------------------------------------------
    # (2) OPTRACE: confirm the candidate Linear.forward is ENTERED before torch._scaled_mm.
    # ---------------------------------------------------------------------------------------
    hits = []
    _orig_smm = torch._scaled_mm

    def _trace_smm(*a, **k):
        hits.append("scaled_mm")
        return _orig_smm(*a, **k)

    cls = type(target)
    _of = cls.forward

    def _trace_fwd(self, *a, **k):
        hits.append(f"FORWARD:{cls.__module__}.{cls.__qualname__}")
        return _of(self, *a, **k)

    torch._scaled_mm = _trace_smm
    cls.forward = _trace_fwd
    try:
        run_one_step()
    finally:
        torch._scaled_mm = _orig_smm
        cls.forward = _of
    # EXPECT: "FORWARD:comfy.ops...Linear" appears, and BEFORE the matching "scaled_mm".
    print("TRACE:", hits[:8])

    # ---------------------------------------------------------------------------------------
    # (3) ACTIVATION RANGE: forward pre-hook on first ~10 eligible fp8 Linears for ONE step.
    # ---------------------------------------------------------------------------------------
    seen = {}

    def mk(name):
        def f(mod, inp):
            x = inp[0].detach()
            seen[name] = float(x.float().abs().amax().cpu())
        return f

    handles = []
    for name, m in model.named_modules():
        w = getattr(m, "weight", None)
        if hasattr(w, "_layout_cls") and str(w._layout_cls).startswith("TensorCoreFP8"):
            handles.append(m.register_forward_pre_hook(mk(name)))
            if len(handles) >= 10:
                break
    try:
        run_one_step()
    finally:
        for h in handles:
            h.remove()
    print("ACT |max| per layer:", sorted(seen.items(), key=lambda kv: -kv[1])[:10], " (fp16 max 65504)")
    print("PROBE DONE — fill in docs/superpowers/results/F-results.md 'Task -1' with these four findings.")


if __name__ == "__main__":
    print(__doc__)
    print("This is a HUMAN-RUN probe. Import it inside ComfyUI and call "
          "probe(model, run_one_step=<one-sampler-step>).")
