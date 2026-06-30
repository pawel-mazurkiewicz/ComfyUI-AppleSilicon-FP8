"""DIAGNOSTIC (opt-in, ASFP8_PROFILE=1): attribute MPS GPU time across the hot ops
of a diffusion/decode step, to find what actually dominates wall-clock.

torch.profiler has no MPS activity backend in this PyTorch (ProfilerActivity has no
MPS member), so it cannot break GPU time down by op: on MPS the CPU-side timeline
only measures async *enqueue*, not on-GPU execution. This wraps the hot seams
(attention, linear, matmul, rms_norm, conv, and ComfyUI-GGUF weight dequant) and
times each with torch.mps.synchronize() so the measured seconds are true on-GPU time.

Synchronizing around every op SERIALIZES the GPU and slows the run (~2-4x). That is
the price of attribution; the *relative* shares between buckets are what matter, not
the absolute wall time of a profiled run. Inert unless ASFP8_PROFILE=1, so it is
safe to leave wired in.

Prints a cumulative, sorted breakdown every ASFP8_PROFILE_INTERVAL seconds (default
20 — roughly one step), plus the unmeasured remainder (elementwise / copies / dequant
math not in a wrapped seam). Stats are cumulative, so later dumps show convergence.

Tip: set a low step count for the profiling run so it finishes quickly, and keep the
config you care about (e.g. mtlflashattn ON) so you measure the real path.

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
_rope_wrapped_ids: set = set()
# id() of every original RoPE function object that has been replaced in
# all its occurrences (including aliases). Unlike _gguf_wrapped (boolean,
# single-target), RoPE is multi-target across many model families and alias
# imports; we never set a global "all done" flag. Each id is added when the
# corresponding original function is first found and wrapped; subsequent scans
# skip ids already present, making repeat calls cheap.

# Canonical module-level function names that perform rotary embedding, plus
# known alias names used by Wan 2.2, EchoShot, Mocha, and KJNodes.
# Derived from:
#   grep -rn "def apply_rope\|def rope_apply\|def apply_rotary\|def _ideogram4" \
#     $COMFYUI --include="*.py"
# and cross-referenced against Codex review of actual import patterns.
#
# NOTE: alias names (apply_rope_comfy, rope_apply_z, etc.) are included so
# Pass 1 can discover them if they appear as standalone functions. Aliases
# bound by 'from X import Y as Z' at import time are also caught by Pass 2
# (object-identity scan) even if their name is not listed here.
_ROPE_FN_NAMES = frozenset({
    # Wan 2.2 canonical
    "rope_apply",
    "rope_apply_3d",
    "rope_apply_1d",
    "apply_rotary_emb_split",
    # Wan 2.2 aliases (wanvideo/modules/model.py lines 27-29, called at 441-451, 677-680)
    "apply_rope_comfy",
    "apply_rope_comfy1",
    # Flux / Chroma / Ideogram canonical
    "apply_rope",
    # HunyuanVideo / LTX
    "apply_rotary_emb",
    # ChatGLM-derived
    "apply_rotary_pos_emb",
    # KJNodes Ideogram4 int8/convrot path
    # (comfyui-kjnodes/nodes/model_optimization_nodes.py lines 1967-1992)
    "_ideogram4_apply_rope_lowp",
    # EchoShot (echoshot/echoshot.py, referenced by wanvideo/modules/model.py line 24)
    "rope_apply_z",
    "rope_apply_c",
    "rope_apply_echoshot",
    # Mocha (mocha/nodes.py, referenced by nodes_sampler.py line 968)
    "rope_apply_mocha",
})


def _is_mps(args):
    """True if any positional arg is a tensor on the MPS device. Non-tensors return
    no .device and are skipped, so list/int/None args are harmless."""
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
    """ComfyUI-GGUF loads AFTER us (custom-node import order), so its dequant cannot
    be wrapped at install() time. Lazily wrap it the first time a real op fires — by
    then GGUF is imported. `ops.py` does `from .dequant import dequantize_tensor`, so
    the live binding lives in the ops module's namespace; patch every module that
    holds the symbol to be safe."""
    global _gguf_wrapped
    if _gguf_wrapped:
        return
    found = False
    for name, mod in list(sys.modules.items()):
        # torch._classes / torch.classes (and similar lazy namespaces) overload
        # __getattr__ so ANY attribute access returns a proxy and probing it raises
        # ("Tried to instantiate class ..."). Skip them and guard every probe.
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
    """Lazily wrap module-level rotary-embedding functions using two-pass identity scanning.

    Called inside every _timed wrapper closure (like _try_wrap_gguf), but unlike
    _try_wrap_gguf (single-target, stops permanently once found), _try_wrap_rope
    never short-circuits globally because:
      1. RoPE spans many model families; new model modules may load late.
      2. Alias imports (from wanvideo.modules.model import rope_apply as apply_rope_comfy1)
         bind the original function object under a different name in the caller's namespace.
         Patching only the source module leaves caller-module aliases stale.

    Algorithm:
      Pass 1 — collect unpatched original function objects by canonical name:
        For each module in sys.modules, look for names in _ROPE_FN_NAMES.
        Skip: _asfp8_timed (already wrapped), no __code__ (builtin/partial),
        id already in _rope_wrapped_ids (wrapped in a prior scan).
        Collect into originals: dict[id(fn) -> fn].

      Pass 2 — patch every occurrence in every module by object identity:
        For every callable value in every module dict whose id() is in originals
        and is not _asfp8_timed, replace it with _timed('rotary', val) and
        record id(val) in _rope_wrapped_ids.
        This catches aliases regardless of their local attribute name.

    If Pass 1 finds no new originals (all known ids already in _rope_wrapped_ids),
    returns immediately — cost is one dict-intersection check per call.
    """
    global _rope_wrapped_ids

    # Pass 1: discover unpatched rope function objects by name
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
                # Skip C-extension callables and builtins — they have no __code__.
                # Note: do NOT check co_argcount; *args-only functions have
                # co_argcount == 0 but are valid RoPE implementations.
                if getattr(fn, "__code__", None) is None:
                    continue
                fn_id = id(fn)
                if fn_id not in _rope_wrapped_ids:
                    originals[fn_id] = fn
        except Exception:
            continue

    if not originals:
        return  # nothing new; cheap exit

    # Pass 2: patch every occurrence in every module (catches alias imports)
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
                # Replace with a fresh wrapper pointing at the original fn.
                # Do not check _rope_wrapped_ids here — the same id may appear
                # in multiple modules (aliases); we want to patch all of them.
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

    # Activation ops. All three share the 'activation' bucket.
    # PREREQUISITE: run Task 0 probe first to confirm nn.SiLU/nn.GELU dispatch
    # through F.silu/F.gelu in this PyTorch version. If they bypass F.*, also
    # wrap nn.SiLU.forward and nn.GELU.forward (see Task 0 probe output).
    F.silu = _timed("activation", F.silu)
    F.gelu = _timed("activation", F.gelu)
    F.glu  = _timed("activation", F.glu)

    # Attempt an early lazy scan for RoPE functions — the model may already be
    # partially imported by the time mps_profile.install() runs. The wrapper also
    # calls _try_wrap_rope() on every timed invocation for the late-import case.
    _try_wrap_rope()

    _t_start = time.perf_counter()
    _t_last_dump = _t_start
    _installed = True
    print(f"{TAG} GPU-time profiler active (synchronized; run is slower). "
          f"Seams: attn / linear / conv2d / conv3d / layernorm / rmsnorm / "
          f"matmul / bmm / activation (silu+gelu+glu) / rotary (lazy, identity-scan) / "
          f"gguf_dequant (lazy). Breakdown every {_interval:.0f}s.")
