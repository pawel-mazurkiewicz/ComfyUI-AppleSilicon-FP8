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
          f"(elementwise / copies / gguf math not in a wrapped seam)")


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

    _t_start = time.perf_counter()
    _t_last_dump = _t_start
    _installed = True
    print(f"{TAG} GPU-time profiler active (synchronized; run is slower). "
          f"Cumulative breakdown every {_interval:.0f}s. GGUF dequant wrapped lazily.")
