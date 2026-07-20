"""Patch #21: fused standalone RoPE (apply_rope / apply_rope_split_half) on MPS.

comfy_kitchen's eager apply_rope is bandwidth-bound but spends several ms/call because it runs the
rotation as ~3-7 separate PyTorch launches with fp32 intermediates and (split-half) two
non-contiguous movedim views (see dev/probe_rope_source.py / probe_rope_cost.py). This replaces
it with ONE compile_shader pass: read x once, read the precomputed 2x2 rotation table once, write
out once, fp32 math in registers, store in the input dtype. freqs_cis is the precomputed rotation
matrix per (position, pair) -- NO complex multiply, NO cos/sin computed in-kernel.

Pure elementwise compile_shader -> works on Apple Silicon M1+ (torch>=2.5); NOT gated on Metal
4.1 / M5. DEFAULT ON, gated on compile_shader (ASFP8_ROPE_FAST=off disables); the #21b comfy-func
retarget is opt-in via ASFP8_ROPE_COMFY_RETARGET=1. Never fatal: any compile/shape/dtype/convention
mismatch falls back to the captured original eager apply_rope*. Per-call backend trace in
_backend_events records (fn_name, "kernel"|"fallback", shape) for the spy tests. Patches the four
comfy_kitchen.backends.eager attributes the registry reads via getattr() on every dispatch, and --
as defence-in-depth -- the same four names on comfy_kitchen.backends.eager.rope.
Scope: 4-D [B,H,L,D], fp32 freqs_cis table, batch/head-broadcast table. Anything else -> fallback."""
from __future__ import annotations
import torch

TAG = "[AppleSilicon-FP8/rope-fast]"

_MSL = r"""
#include <metal_stdlib>
using namespace metal;
constant constexpr uint TG = 256;
kernel void apply_rope_fused(
    device const @T@*   x    [[buffer(0)]],
    device const float* fr   [[buffer(1)]],
    device       @T@*   out  [[buffer(2)]],
    device const int*   meta [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]])
{
    const int  D     = meta[0];
    const int  halfD = meta[1];
    const int  L     = meta[2];
    const uint ny    = uint(meta[3]);
    const uint rows  = uint(meta[4]);
    const int  split = meta[5];
    const uint row = tgpig.z * ny + tgpig.y;
    if (row >= rows) return;
    const ulong xbase = ulong(row) * ulong(D);
    const uint  l     = row % uint(L);
    const ulong fbase = ulong(l) * ulong(halfD) * 4ul;
    for (uint p = lid; p < uint(halfD); p += TG) {
        const ulong fo  = fbase + ulong(p) * 4ul;
        const float f00 = fr[fo + 0], f01 = fr[fo + 1];
        const float f10 = fr[fo + 2], f11 = fr[fo + 3];
        uint ia, ib;
        if (split == 0) { ia = 2u*p; ib = 2u*p + 1u; }
        else            { ia = p;    ib = p + uint(halfD); }
        const float a = float(x[xbase + ia]);
        const float b = float(x[xbase + ib]);
        out[xbase + ia] = @T@(f00 * a + f01 * b);
        out[xbase + ib] = @T@(f10 * a + f11 * b);
    }
}
"""

_MSL_T = {torch.float16: "half", torch.bfloat16: "bfloat", torch.float32: "float"}
_TG = 256
_NY = 32768
_libs: dict[str, object] = {}
_meta_cache: dict[tuple, torch.Tensor] = {}
_warned = False

# Per-call backend trace (MAJOR 5). Each kernel/fallback decision appends one event.
# Tests inspect the tail to assert BOTH calls of a pair wrapper took the intended path.
_backend_events: list[tuple] = []   # (fn_name, "kernel"|"fallback", tuple(shape))

# captured originals (set in install()/install_for_test()); fallback uses these.
_orig: dict = {}                    # name -> original eager fn


def _record(name, backend, shape):
    _backend_events.append((name, backend, tuple(shape)))


def _last_backend():
    """Convenience for single-call tests: backend of the most recent event."""
    return _backend_events[-1][1] if _backend_events else "fallback"


def _get_lib(dtype):
    t = _MSL_T[dtype]
    lib = _libs.get(t)
    if lib is None:
        lib = torch.mps.compile_shader(_MSL.replace("@T@", t))
        _libs[t] = lib
    return lib


def _meta(D, halfD, L, ny, rows, split, device):
    key = (D, halfD, L, ny, rows, split, str(device))
    t = _meta_cache.get(key)
    if t is None:
        t = torch.tensor([D, halfD, L, ny, rows, split], dtype=torch.int32, device=device)
        _meta_cache[key] = t
    return t


def _reference_interleaved(x, freqs_cis):
    """Eager-FORMULA reference for apply_rope1 (interleaved). DIAGNOSTIC ONLY -- not the
    primary oracle. The primary oracle in tests is the captured REAL eager op (see Task 2)."""
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    f = freqs_cis
    if x_.shape[2] != 1 and f.shape[2] != 1 and x_.shape[2] != f.shape[2]:
        f = f[:, :, :x_.shape[2]]
    out = f[..., 0] * x_[..., 0]
    out = out + f[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).type_as(x)


def _reference_split_half(x, freqs_cis):
    """Eager-FORMULA reference for apply_rope_split_half1. DIAGNOSTIC ONLY."""
    t_ = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs_cis.dtype)
    t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
    return t_out.movedim(-1, -2).reshape(*x.shape).type_as(x)


def _fallback1(name, x, freqs_cis):
    _record(name, "fallback", x.shape)
    return _orig[name](x, freqs_cis)


def _prep_table(freqs_cis, halfD, L):
    """Return contiguous fp32 [Lf,halfD,4] table or None (-> fallback).

    No cache (MAJOR 9 contingency): a ``weakref.WeakKeyDictionary`` keyed by a tensor crashes on
    this torch build because ``weakref.ref.__eq__`` delegates to the tensor's elementwise ``__eq__``
    ("Boolean value of Tensor with more than one value is ambiguous"). The plan's sanctioned
    fallback is to recompute ``fr`` every call -- which is free for the common case: an
    already-contiguous fp32 table makes ``reshape`` a metadata-only view and ``contiguous`` a no-op
    (returns the same storage, no copy). Recomputing also means the table can never go stale, so
    in-place table mutation is reflected automatically."""
    if freqs_cis.dtype != torch.float32:                # MAJOR 8: hard-fallback on non-fp32
        return None
    fs = freqs_cis.shape
    if len(fs) < 4 or fs[-1] != 2 or fs[-2] != 2 or fs[-3] != halfD:
        return None
    Lf = fs[-4]
    if Lf < L:
        return None
    if any(int(d) != 1 for d in fs[:-4]):               # require batch/head broadcast (Open Q #2)
        return None
    return freqs_cis.reshape(Lf, halfD, 4).contiguous()  # already fp32; view + no-op for contiguous


def _launch(x, fr, D, halfD, L, split):
    """Low-level kernel launch. ``x`` must be rank-4 MPS with a supported dtype; ``fr`` a contiguous
    fp32 rotation table whose rows are indexed by ``row % L`` (table layout [Lf, halfD, 4], flattened
    f00,f01,f10,f11 per pair). Returns ``out`` shaped like ``x``. Raises on any kernel failure (the
    caller turns that into a fallback)."""
    xc = x.contiguous()
    rows = xc.numel() // D
    ny = min(rows, _NY)
    nz = (rows + ny - 1) // ny
    out = torch.empty_like(xc)
    meta = _meta(D, halfD, L, ny, rows, split, x.device)
    _get_lib(x.dtype).apply_rope_fused(
        xc, fr, out, meta, threads=(_TG, ny, nz), group_size=(_TG, 1, 1))
    return out.view_as(x)


def _rope_fused(name, x, freqs_cis, split):
    global _warned
    if (x.device.type != "mps" or x.dtype not in _MSL_T
            or freqs_cis.device != x.device or x.numel() == 0 or x.dim() != 4):  # MAJOR 6: rank-4 only
        return _fallback1(name, x, freqs_cis)
    try:
        D = x.shape[-1]
        if D % 2 != 0:
            return _fallback1(name, x, freqs_cis)
        halfD = D // 2
        L = x.shape[-2]
        fr = _prep_table(freqs_cis, halfD, L)
        if fr is None:
            return _fallback1(name, x, freqs_cis)
        out = _launch(x, fr, D, halfD, L, split)
        _record(name, "kernel", x.shape)
        return out
    except Exception as e:
        if not _warned:
            print(f"{TAG} kernel error ({e}); falling back to eager rope.", flush=True)
            _warned = True
        return _fallback1(name, x, freqs_cis)


def apply_rope1_fused(x, freqs_cis):
    return _rope_fused("apply_rope1", x, freqs_cis, split=0)


def apply_rope_split_half1_fused(x, freqs_cis):
    return _rope_fused("apply_rope_split_half1", x, freqs_cis, split=1)


def apply_rope_fused_pair(xq, xk, freqs_cis):
    return apply_rope1_fused(xq, freqs_cis), apply_rope1_fused(xk, freqs_cis)


def apply_rope_split_half_fused_pair(xq, xk, freqs_cis):
    return (apply_rope_split_half1_fused(xq, freqs_cis),
            apply_rope_split_half1_fused(xk, freqs_cis))


# ===========================================================================
# Patch #21b: retarget the REAL comfy RoPE functions the live models call.
#
# A live probe (2026-07-01) showed comfy_kitchen.backends.eager fires at most ONCE per model --
# the DiT / TE actually call comfy's OWN rope functions, which we must intercept directly:
#
#   comfy.ldm.flux.math.apply_rope / .apply_rope1      INTERLEAVED 2x2 form (single fp32 freqs
#       tensor, shape (...,halfD,2,2)); identical convention to comfy_kitchen's apply_rope, so it
#       maps onto the existing split=0 kernel with NO new math.  (flux.math.apply_rope itself only
#       delegates to comfy.quant_ops.ck.apply_rope when not training; we intercept ABOVE that.)
#
#   comfy.text_encoders.llama.apply_rope               HALF-SPLIT rotate_half form. freqs_cis is a
#   (== comfy.ldm.ideogram4.model.apply_rope,           3-TUPLE (cos, sin, nsin), cos full-dim D,
#       same object, imported by alias)                 sin/nsin half-dim. Expressed on the split=1
#       kernel by building a per-pair 2x2 table:
#           out[p]        = cos[p]*x[p]      + nsin[p]*x[p+halfD]   (= f00*a + f01*b)
#           out[p+halfD]  = sin[p]*x[p]      + cos[p+halfD]*x[p+halfD] (= f10*a + f11*b)
#       i.e. table row = [cos[:halfD], nsin, sin, cos[halfD:]]. Bit-faithful to the real fn (it does
#       q*cos then addcmul_ with nsin/sin in fp32, then casts back).
#
# Both retargets are guarded, never fatal, and gated on the SAME opt-in ASFP8_ROPE_FAST=1.
# Fallback for each comfy target is the captured REAL comfy function (NOT eager) -- so a fallback is
# always bit-identical to stock comfy. Anything the kernel cannot match -> fall back, never wrong.
# ===========================================================================

# captured REAL comfy originals (oracle + fallback). Keys: 'flux.apply_rope', 'flux.apply_rope1',
# 'llama.apply_rope'. Kept separate from _orig (which holds comfy_kitchen eager originals).
_orig_comfy: dict = {}
_comfy_replacements: list = []   # (module_dict, attr_name, original_fn) for clean uninstall


def _interleaved_single(x, freqs_cis):
    """Run the split=0 (interleaved 2x2) kernel for one tensor. Returns out or None (-> fallback).
    Shared validity gate with _rope_fused, but never touches _orig / never records."""
    if (x.device.type != "mps" or x.dtype not in _MSL_T
            or not torch.is_tensor(freqs_cis) or freqs_cis.device != x.device
            or x.numel() == 0 or x.dim() != 4):
        return None
    D = x.shape[-1]
    if D % 2 != 0:
        return None
    halfD = D // 2
    L = x.shape[-2]
    fr = _prep_table(freqs_cis, halfD, L)
    if fr is None:
        return None
    return _launch(x, fr, D, halfD, L, split=0)


def _flux_apply_rope1_fused(x, freqs_cis):
    global _warned
    try:
        out = _interleaved_single(x, freqs_cis)
    except Exception as e:
        if not _warned:
            print(f"{TAG} flux apply_rope1 kernel error ({e}); falling back.", flush=True)
            _warned = True
        out = None
    if out is None:
        _record("flux.apply_rope1", "fallback", getattr(x, "shape", ()))
        return _orig_comfy["flux.apply_rope1"](x, freqs_cis)
    _record("flux.apply_rope1", "kernel", x.shape)
    return out


def _flux_apply_rope_fused(xq, xk, freqs_cis):
    """flux pair wrapper. Run the kernel for BOTH q and k; if either cannot, fall back the whole
    pair to the real comfy original (never mix kernel + fallback within one pair)."""
    global _warned
    try:
        oq = _interleaved_single(xq, freqs_cis)
        ok = _interleaved_single(xk, freqs_cis) if oq is not None else None
    except Exception as e:
        if not _warned:
            print(f"{TAG} flux apply_rope kernel error ({e}); falling back.", flush=True)
            _warned = True
        oq = ok = None
    if oq is None or ok is None:
        _record("flux.apply_rope", "fallback", getattr(xq, "shape", ()))
        return _orig_comfy["flux.apply_rope"](xq, xk, freqs_cis)
    _record("flux.apply_rope", "kernel", xq.shape)
    _record("flux.apply_rope", "kernel", xk.shape)
    return oq, ok


def _prep_table_llama(freqs_cis, halfD):
    """Build a contiguous fp32 [L, halfD, 4] split=1 rotation table from llama's (cos, sin, nsin)
    3-tuple, or return (None, 0) -> fallback. cos has last-dim 2*halfD; sin/nsin have last-dim halfD.
    Leading dims must collapse to exactly L rows (i.e. batch/head broadcast == 1) so the kernel's
    row%L indexing is valid; a true batch (>1) table is rejected (-> fallback)."""
    if not (isinstance(freqs_cis, (tuple, list)) and len(freqs_cis) == 3):
        return None, 0
    cos, sin, nsin = freqs_cis
    if not all(torch.is_tensor(t) for t in (cos, sin, nsin)):
        return None, 0
    if cos.shape[-1] != 2 * halfD or sin.shape[-1] != halfD or nsin.shape[-1] != halfD:
        return None, 0

    def _flatten(t, last):
        flat = t.reshape(-1, last)   # collapse all leading dims
        return flat                  # rows must equal L (checked against the cos rows below)

    cflat = _flatten(cos, 2 * halfD)
    sflat = _flatten(sin, halfD)
    nflat = _flatten(nsin, halfD)
    L = cflat.shape[0]
    if sflat.shape[0] != L or nflat.shape[0] != L:   # batch/head broadcast mismatch -> fallback
        return None, 0
    cos_lo = cflat[:, :halfD]
    cos_hi = cflat[:, halfD:]
    table = torch.stack([cos_lo, nflat, sflat, cos_hi], dim=-1)   # (L, halfD, 4)
    return table.reshape(L, halfD, 4).to(torch.float32).contiguous(), L


def _llama_apply_rope_fused(xq, xk, freqs_cis):
    """llama/ideogram pair wrapper (half-split rotate_half). Build the 2x2 table once from
    (cos,sin,nsin), then run the split=1 kernel for both q and k. Fall back the whole pair to the
    real comfy original on any mismatch."""
    global _warned
    fb = _orig_comfy["llama.apply_rope"]
    try:
        if (xq.device.type != "mps" or xq.dtype not in _MSL_T or xk.dtype not in _MSL_T
                or xq.dim() != 4 or xk.dim() != 4
                or xq.shape[-1] % 2 or xk.shape[-1] != xq.shape[-1]
                or xq.numel() == 0 or xk.numel() == 0):
            raise ValueError("unsupported shape/dtype")
        D = xq.shape[-1]
        halfD = D // 2
        table, L = _prep_table_llama(freqs_cis, halfD)
        if table is None or table.device != xq.device:
            raise ValueError("table not buildable on device")
        # The table has exactly L position rows; require both tensors' sequence length == L so the
        # kernel's row%L indexing lands on the right position (self-attention: Lq == Lk == L).
        if xq.shape[-2] != L or xk.shape[-2] != L:
            raise ValueError("seq length != table rows")
        oq = _launch(xq, table, D, halfD, L, split=1)
        ok = _launch(xk, table, D, halfD, L, split=1)
    except Exception as e:
        if not isinstance(e, ValueError) and not _warned:
            print(f"{TAG} llama apply_rope kernel error ({e}); falling back.", flush=True)
            _warned = True
        _record("llama.apply_rope", "fallback", getattr(xq, "shape", ()))
        return fb(xq, xk, freqs_cis)
    _record("llama.apply_rope", "kernel", xq.shape)
    _record("llama.apply_rope", "kernel", xk.shape)
    return oq, ok


def _install_comfy():
    """Capture the REAL comfy rope functions and replace every reference to them (by object
    identity, mirroring mps_profile._try_wrap_rope) so aliased re-imports -- e.g.
    comfy.ldm.ideogram4.model.apply_rope, which IS llama.apply_rope -- are all rerouted. Best-effort:
    a missing module just skips that target."""
    import sys
    id_map = {}   # id(original_fn) -> wrapper
    try:
        import comfy.ldm.flux.math as _fm
        _orig_comfy["flux.apply_rope"] = _fm.apply_rope
        _orig_comfy["flux.apply_rope1"] = _fm.apply_rope1
        id_map[id(_fm.apply_rope)] = _flux_apply_rope_fused
        id_map[id(_fm.apply_rope1)] = _flux_apply_rope1_fused
    except Exception:
        pass
    try:
        import comfy.text_encoders.llama as _ll
        _orig_comfy["llama.apply_rope"] = _ll.apply_rope
        id_map[id(_ll.apply_rope)] = _llama_apply_rope_fused
    except Exception:
        pass
    if not id_map:
        return
    for _mod_name, mod in list(sys.modules.items()):
        if (mod is None or _mod_name.startswith("torch._classes")
                or _mod_name.startswith("torch.classes")):
            continue
        try:
            d = mod.__dict__
            for attr, val in list(d.items()):
                wrap = id_map.get(id(val))
                if wrap is not None and val is not wrap:
                    _comfy_replacements.append((d, attr, val))
                    d[attr] = wrap
        except Exception:
            continue


def _uninstall_comfy():
    for d, attr, orig in _comfy_replacements:
        try:
            d[attr] = orig
        except Exception:
            pass
    _comfy_replacements.clear()
    _orig_comfy.clear()


_PATCH_MAP = {
    "apply_rope1": "apply_rope1_fused",
    "apply_rope": "apply_rope_fused_pair",
    "apply_rope_split_half1": "apply_rope_split_half1_fused",
    "apply_rope_split_half": "apply_rope_split_half_fused_pair",
}
_installed = False


def _targets():
    """The module objects whose attrs we patch: the eager PACKAGE (read by the registry) and the
    eager.rope SOURCE module (defence-in-depth for direct importers / pair-wrapper internals)."""
    import comfy_kitchen.backends.eager as e
    import comfy_kitchen.backends.eager.rope as r
    return (e, r)


def _do_install():
    global _installed
    if _installed:
        return True
    e, r = _targets()
    for name in _PATCH_MAP:
        _orig[name] = getattr(e, name)        # capture original (used by fallback + as oracle)
    g = globals()
    for name, repl in _PATCH_MAP.items():
        setattr(e, name, g[repl])             # the attr the registry reads via getattr
        try:
            setattr(r, name, g[repl])         # source-module global (MAJOR 4)
        except Exception:
            pass
    _installed = True
    return True


def install():
    # Patch #21 (standalone fused RoPE kernel) is DEFAULT ON, gated on compile_shader
    # (Tier A: fp32 math, no Metal-4.1/M5 needed). ASFP8_ROPE_FAST=off disables; =1 forces on.
    from . import _caps
    if not _caps.resolve("ASFP8_ROPE_FAST", default_on=True, cap=_caps.has_compile_shader):
        return
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        return
    have_ck = True
    try:
        import comfy_kitchen.backends.eager  # noqa: F401
    except Exception:
        have_ck = False
        print(f"{TAG} comfy_kitchen not installed; eager-rope retarget off.", flush=True)
    if have_ck:
        try:
            _do_install()
        except Exception as e:                 # never fatal
            print(f"{TAG} eager install failed ({e}); leaving eager rope untouched.", flush=True)
    # Patch #21b: retarget the REAL comfy rope functions the live models call
    # (comfy.ldm.flux / llama apply_rope). Higher-risk — it patches live model code — so it
    # stays OPT-IN behind its own flag (default OFF) until it has broader cross-model
    # validation. ASFP8_ROPE_COMFY_RETARGET=1 turns it on. Independent of comfy_kitchen; never fatal.
    if _caps.resolve("ASFP8_ROPE_COMFY_RETARGET", default_on=False, cap=_caps.has_compile_shader):
        try:
            _install_comfy()
        except Exception as e:
            print(f"{TAG} comfy retarget failed ({e}); leaving comfy rope untouched.", flush=True)
    targeted = sorted(_orig_comfy.keys())
    print(f"{TAG} fused RoPE active on MPS (eager apply_rope/apply_rope_split_half"
          f"{' + comfy ' + str(targeted) if targeted else ''} rerouted; interleaved + split-half; "
          f"fp32 math, no Metal-4.1/M5 requirement; comfy-retarget "
          f"{'ON' if targeted else 'OFF (ASFP8_ROPE_COMFY_RETARGET=1 to enable)'}).", flush=True)


def install_for_test():
    _do_install()


def install_comfy_for_test():
    """Test-only: capture + patch the comfy targets (requires comfy importable)."""
    _install_comfy()


def uninstall_comfy_for_test():
    _uninstall_comfy()


def uninstall_for_test():
    global _installed
    if not _installed:
        return
    e, r = _targets()
    for name, fn in _orig.items():
        setattr(e, name, fn)
        try:
            setattr(r, name, fn)
        except Exception:
            pass
    _orig.clear()
    _installed = False
