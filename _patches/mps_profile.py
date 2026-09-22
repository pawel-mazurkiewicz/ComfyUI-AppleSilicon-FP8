"""DIAGNOSTIC (opt-in, ASFP8_PROFILE=1): attribute MPS GPU time across the hot ops.

torch.profiler has no MPS activity backend, so the CPU timeline only measures async
enqueue. This times the hot seams around torch.mps.synchronize() instead, which
serializes the GPU and slows the run ~2-4x — the relative shares are what matter.

  ASFP8_PROFILE=1                 enable
  ASFP8_PROFILE_INTERVAL=20       seconds between cumulative dumps
"""

import os
import sys
import time

import torch

TAG = "[AppleSilicon-FP8/mps_profile]"

_installed = False
_stats = {}            # name -> [calls, total_seconds]
_t_start = None
_t_last_dump = None
_interval = 20.0
_gguf_wrapped = False
# id() of every original RoPE function already replaced in all its occurrences.
# Never latched globally: RoPE is multi-target and model modules can load late.
_rope_wrapped_ids: set = set()

# module-level rotary-embedding function names, canonical and known aliases
_ROPE_FN_NAMES = frozenset({
    # Wan 2.2
    "rope_apply",
    "rope_apply_3d",
    "rope_apply_1d",
    "apply_rotary_emb_split",
    "apply_rope_comfy",
    "apply_rope_comfy1",
    # Flux / Chroma / Ideogram
    "apply_rope",
    # HunyuanVideo / LTX
    "apply_rotary_emb",
    # ChatGLM-derived
    "apply_rotary_pos_emb",
    # KJNodes Ideogram4 int8/convrot
    "_ideogram4_apply_rope_lowp",
    # EchoShot
    "rope_apply_z",
    "rope_apply_c",
    "rope_apply_echoshot",
    # Mocha
    "rope_apply_mocha",
})


def _is_mps(args):
    """True if any positional arg is a tensor on the MPS device."""
    for a in args:
        d = getattr(a, "device", None)
        if d is not None and getattr(d, "type", None) == "mps":
            return True
    return False


def _record(name, dt):
    s = _stats.get(name)
    if s is None:
        s = [0, 0.0]
        _stats[name] = s
    s[0] += 1
    s[1] += dt


def _dump():
    now = time.perf_counter()
    elapsed = now - _t_start
    measured = sum(v[1] for v in _stats.values())
    other = max(0.0, elapsed - measured)
    print(f"{TAG} GPU-time breakdown after {elapsed:.1f}s wall "
          f"({measured:.1f}s in wrapped ops, {other:.1f}s elsewhere):")
    rows = sorted(_stats.items(), key=lambda kv: -kv[1][1])
    for name, (calls, total) in rows:
        pct = (100.0 * total / elapsed) if elapsed else 0.0
        avg_ms = (1000.0 * total / calls) if calls else 0.0
        print(f"{TAG}   {name:<14} {total:8.2f}s  {pct:5.1f}%  "
              f"({calls} calls, {avg_ms:.3f} ms/call)")
    print(f"{TAG}   {'<unwrapped>':<14} {other:8.2f}s  "
          f"{(100.0 * other / elapsed) if elapsed else 0.0:5.1f}%  "
          f"(elementwise / modulation / rotary-if-not-found / "
          f"gguf-if-not-found / copies / misc not in a wrapped seam)")


def _maybe_dump():
    global _t_last_dump
    now = time.perf_counter()
    if now - _t_last_dump >= _interval:
        _t_last_dump = now
        _dump()


def _timed(name, orig):
    def wrapper(*args, **kwargs):
        if not _is_mps(args):
            return orig(*args, **kwargs)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        out = orig(*args, **kwargs)
        torch.mps.synchronize()
        _record(name, time.perf_counter() - t0)
        _try_wrap_gguf()
        _try_wrap_rope()
        _maybe_dump()
        return out
    wrapper._asfp8_timed = True
    return wrapper


def _try_wrap_gguf():
    """Wrap ComfyUI-GGUF's dequant lazily: it imports after us, and its live binding
    sits in the importing module's namespace, so patch every module holding the name."""
    global _gguf_wrapped
    if _gguf_wrapped:
        return
    found = False
    for name, mod in list(sys.modules.items()):
        # torch._classes overloads __getattr__, so any attribute probe raises
        if mod is None or name.startswith("torch._classes") or name.startswith("torch.classes"):
            continue
        try:
            fn = mod.__dict__.get("dequantize_tensor")  # dict, not getattr: no __getattr__ magic
            if not callable(fn) or getattr(fn, "_asfp8_timed", False):
                continue
            if "dequant" in getattr(fn, "__module__", ""):
                mod.dequantize_tensor = _timed("gguf_dequant", fn)
                found = True
        except Exception:
            continue
    if found:
        _gguf_wrapped = True


def _try_wrap_rope():
    """Lazily wrap module-level rotary-embedding functions, by name then by identity.

    The second pass is what catches `from X import rope_apply as apply_rope_comfy1`:
    patching only the defining module would leave every caller's alias stale.
    """
    global _rope_wrapped_ids

    # pass 1: discover unpatched rope function objects by name
    originals: dict = {}  # id(fn) -> fn
    for _mod_name, mod in list(sys.modules.items()):
        if (mod is None
                or _mod_name.startswith("torch._classes")
                or _mod_name.startswith("torch.classes")):
            continue
        try:
            mod_dict = mod.__dict__
            for fn_name in _ROPE_FN_NAMES:
                fn = mod_dict.get(fn_name)
                if fn is None or not callable(fn):
                    continue
                if getattr(fn, "_asfp8_timed", False):
                    continue
                # skip builtins; do NOT also check co_argcount, *args-only RoPE
                # implementations have co_argcount == 0
                if getattr(fn, "__code__", None) is None:
                    continue
                fn_id = id(fn)
                if fn_id not in _rope_wrapped_ids:
                    originals[fn_id] = fn
        except Exception:
            continue

    if not originals:
        return

    # pass 2: patch every occurrence by identity, which catches alias imports
    for _mod_name, mod in list(sys.modules.items()):
        if (mod is None
                or _mod_name.startswith("torch._classes")
                or _mod_name.startswith("torch.classes")):
            continue
        try:
            for attr_name, val in list(mod.__dict__.items()):
                if not callable(val):
                    continue
                if getattr(val, "_asfp8_timed", False):
                    continue
                fn_id = id(val)
                if fn_id not in originals:
                    continue
                # deliberately not checking _rope_wrapped_ids: one id can appear in
                # several modules as an alias, and all of them need patching
                _rope_wrapped_ids.add(fn_id)
                mod.__dict__[attr_name] = _timed("rotary", originals[fn_id])
        except Exception:
            continue


def install():
    global _installed, _t_start, _t_last_dump, _interval
    if _installed:
        return
    if os.environ.get("ASFP8_PROFILE") != "1":
        return
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        _interval = float(os.environ.get("ASFP8_PROFILE_INTERVAL", "20"))
    except ValueError:
        _interval = 20.0

    import torch.nn.functional as F

    F.scaled_dot_product_attention = _timed("attn", F.scaled_dot_product_attention)
    F.linear = _timed("linear", F.linear)
    F.conv2d = _timed("conv2d", F.conv2d)
    F.conv3d = _timed("conv3d", F.conv3d)
    F.layer_norm = _timed("layernorm", F.layer_norm)
    if hasattr(F, "rms_norm"):
        F.rms_norm = _timed("rmsnorm", F.rms_norm)
    torch.matmul = _timed("matmul", torch.matmul)
    torch.bmm = _timed("bmm", torch.bmm)

    # all three share the 'activation' bucket
    F.silu = _timed("activation", F.silu)
    F.gelu = _timed("activation", F.gelu)
    F.glu  = _timed("activation", F.glu)

    # the model may already be partially imported; _timed rescans for late imports
    _try_wrap_rope()

    _t_start = time.perf_counter()
    _t_last_dump = _t_start
    _installed = True
    print(f"{TAG} GPU-time profiler active (synchronized; run is slower). "
          f"Seams: attn / linear / conv2d / conv3d / layernorm / rmsnorm / "
          f"matmul / bmm / activation (silu+gelu+glu) / rotary (lazy, identity-scan) / "
          f"gguf_dequant (lazy). Breakdown every {_interval:.0f}s.")
