# dev/probe_rope_runtime.py — import-once ORIGIN logger. HUMAN runs a few steps of Flux-2/Ideogram-4.
# Answers BLOCKER 1: is the measured 'rotary' bucket the comfy_kitchen custom-op path, or a
# non-comfy_kitchen rotary (e.g. KJNodes _ideogram4_apply_rope_lowp)?
import sys, torch
_seen = set()

# Note: torch.ops.comfy_kitchen.* entries are not reassignable, so we cannot wrap them directly.
# Instead we wrap the EAGER attrs below: the custom op dispatches to eager via
# registry.get_implementation, so "an eager attr fired" IS proof the comfy_kitchen path executed.
# is_comfy_kitchen is derived from each fired function's __module__.

def _wrap(label, fn):
    def w(*a, **k):
        key = label + ":" + getattr(fn, "__qualname__", "?")
        if key not in _seen:
            _seen.add(key)
            t = a[0] if a else None
            f = k.get("freqs_cis", a[-1] if a else None)
            print(f"[ROPE-ORIGIN] fired label={label} "
                  f"module={getattr(fn,'__module__','?')} qualname={getattr(fn,'__qualname__','?')} "
                  f"id={id(fn)} is_comfy_kitchen={str(getattr(fn,'__module__','')).startswith('comfy_kitchen')}",
                  flush=True)
            try:
                print(f"             x.rank={t.dim()} x.shape={tuple(t.shape)} x.dtype={t.dtype} "
                      f"x.contig={t.is_contiguous()} freqs.rank={f.dim()} freqs.shape={tuple(f.shape)} "
                      f"freqs.dtype={f.dtype}", flush=True)
            except Exception:
                pass
        return fn(*a, **k)
    return w

# (a) comfy_kitchen eager attrs — firing here proves the comfy_kitchen path.
import comfy_kitchen.backends.eager as e
for n in ("apply_rope", "apply_rope1", "apply_rope_split_half", "apply_rope_split_half1"):
    setattr(e, n, _wrap("comfy_kitchen.eager", getattr(e, n)))

# (b) non-comfy_kitchen rotary names across already-imported modules (KJNodes etc.).
_NAMES = ("apply_rope", "apply_rotary_emb", "_ideogram4_apply_rope_lowp",
          "rope_apply", "apply_rotary_pos_emb")
for _mod_name, _mod in list(sys.modules.items()):
    if _mod is None or _mod_name.startswith("comfy_kitchen"):
        continue
    for _fn_name in _NAMES:
        _fn = getattr(_mod, _fn_name, None)
        if callable(_fn) and getattr(_fn, "__code__", None) is not None:
            try:
                setattr(_mod, _fn_name, _wrap(f"{_mod_name}.{_fn_name}", _fn))
            except Exception:
                pass

print("[ROPE-ORIGIN] installed; run one Flux-2 and one Ideogram-4 render, then read which "
      "label(s) fire and whether is_comfy_kitchen=True.", flush=True)
