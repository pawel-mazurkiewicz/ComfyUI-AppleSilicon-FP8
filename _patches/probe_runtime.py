"""DIAGNOSTIC (opt-in, ASFP8_PROBE=1): log the fp8 Linear seam/scale/range and the
rotary origin during a normal render.

Read-only and never fatal; each finding logs once. Run with the optimizations OFF so the
probe sees the unmodified model, then read the [F-PROBE] and [ROPE-ORIGIN] lines.
"""

import os

TAG = "[F-PROBE]"

_installed = False
_seam_logged = set()          # class keys already dumped (+ the sentinel "ORDER")
_smm = {"fired": False, "inside_fp8_fwd": False}
_fp8_depth = 0


def _is_fp8_weight(w):
    return hasattr(w, "_layout_cls") and str(getattr(w, "_layout_cls", "")).startswith("TensorCoreFP8")


def _install_rope_origin():
    """Load dev/probe_rope_runtime.py by path; it self-installs its loggers on import."""
    import os.path as p
    import importlib.util as u
    repo_root = p.dirname(p.dirname(p.abspath(__file__)))
    script = p.join(repo_root, "dev", "probe_rope_runtime.py")
    if not p.exists(script):
        print(f"{TAG} rope-origin probe not found at {script}")
        return
    spec = u.spec_from_file_location("_asfp8_rope_origin_probe", script)
    mod = u.module_from_spec(spec)
    spec.loader.exec_module(mod)


def _install_fp8_seam(torch):
    """Forward hooks that dump the fp8 seam/scale/range on the first fp8 layer, plus
    whether torch._scaled_mm fires inside that forward."""
    from torch.nn.modules.module import register_module_forward_hook, register_module_forward_pre_hook

    _orig_smm = torch._scaled_mm

    def _smm_wrap(*a, **k):
        _smm["fired"] = True
        if _fp8_depth > 0:
            _smm["inside_fp8_fwd"] = True
        return _orig_smm(*a, **k)

    torch._scaled_mm = _smm_wrap

    def _pre(module, inp):
        global _fp8_depth
        w = getattr(module, "weight", None)
        if not _is_fp8_weight(w):
            return
        _fp8_depth += 1
        cls = type(module)
        key = f"{cls.__module__}.{cls.__qualname__}"
        if key in _seam_logged:
            return
        _seam_logged.add(key)
        try:
            sc = w._params.scale
            sc_shape = tuple(sc.shape) if hasattr(sc, "shape") else "scalar"
            x = inp[0] if inp else None
            amax = float(x.detach().float().abs().amax().cpu()) if x is not None else -1.0
            print(f"{TAG} SEAM  class={key}")
            print(f"{TAG}   MRO   ={[c.__name__ for c in cls.__mro__]}")
            print(f"{TAG}   layout={w._layout_cls}  qdata={tuple(w._qdata.shape)},{w._qdata.dtype}  "
                  f"scale={sc_shape},numel={sc.numel()},{sc.dtype}  "
                  f"-> {'PER-CHANNEL [N]' if sc.numel() > 1 else 'SCALAR'}")
            print(f"{TAG}   ACT|max|={amax:.2f} (fp16 max 65504) -> "
                  f"{'RANGE OK (guard off)' if 0 <= amax < 65504 else 'RANGE GUARD NEEDED'}")
        except Exception as e:
            print(f"{TAG}   seam dump error {e!r}")

    def _post(module, inp, out):
        global _fp8_depth
        w = getattr(module, "weight", None)
        if not _is_fp8_weight(w) or _fp8_depth <= 0:
            return
        _fp8_depth -= 1
        if _smm["fired"] and "ORDER" not in _seam_logged:
            _seam_logged.add("ORDER")
            verdict = "BEFORE (forward wraps _scaled_mm -> seam interceptable)" \
                if _smm["inside_fp8_fwd"] else "NOT before (activation quantized elsewhere -> seam SUSPECT)"
            print(f"{TAG} ORDER fp8 Linear.forward ran {verdict}")

    register_module_forward_pre_hook(_pre)
    register_module_forward_hook(_post)


def install():
    global _installed
    if _installed:
        return
    if os.environ.get("ASFP8_PROBE", "").strip().lower() not in ("1", "on", "true"):
        return
    import sys
    if sys.platform != "darwin":
        return
    try:
        import torch
    except Exception:
        return
    _installed = True
    try:
        _install_rope_origin()
    except Exception as e:
        print(f"{TAG} rope-origin probe skipped ({e!r})")
    try:
        _install_fp8_seam(torch)
        print(f"{TAG} fp8 seam/scale/range auto-probe armed. Run ONE Flux-2 render with the "
              f"optimizations OFF; read the [F-PROBE ...] lines.")
    except Exception as e:
        print(f"{TAG} fp8 seam probe skipped ({e!r})")
