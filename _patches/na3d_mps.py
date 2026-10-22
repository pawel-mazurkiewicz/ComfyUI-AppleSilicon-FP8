"""Patch: fused 3D neighborhood attention (comfy_kitchen.na3d) on MPS.

The LTX 2.4/2.5 diffusion VAE decoder runs NATTEN-style na3d (11x11x11 windows at stage 5)
over every pixel patch, and comfy_kitchen has no MPS backend for it: the eager fallback
stacks dense masked SDPA calls per tile, ~80% of a decode. Here one threadgroup takes a
4x4x4 brick of queries for one head and gathers the union of their windows into
threadgroup memory 64 keys at a time; each simdgroup owns one frame of the brick and runs
S = QK^T and O += PV as matmul2d on the tensor units, with its rows' windows applied as a
mask in an online softmax held in registers, skipping key blocks outside its frames.
Non-causal, head_dim 64, bf16/fp16 (each dtype self-checked); anything else, or any grad,
runs the original. Only the `comfy_kitchen.na3d` attribute is replaced. DEFAULT ON where
the shader builds and passes its self-check; ASFP8_NA3D=off disables.

Ported from Illuminus7/ComfyUI-AppleSilicon-int8-Fast (MIT).
"""

import sys

import torch

from . import _caps

TAG = "[AppleSilicon-FP8/na3d]"

_installed = False
_lib = None
_orig = None
_good = set()   # dtypes whose self-check passed

_BRICK = 4      # QT/QH/QW in the shader
_MAX_EXT = 255  # window and brick-union extents are packed in uchar4

_MSL = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// a QT x QH x QW query brick per threadgroup, simdgroup s owning frame t0 + s (SGQ rows)
constant constexpr int QT = 4, QH = 4, QW = 4, SGQ = QH * QW, BKV = 64, D = 64;
constant constexpr int CAP = SGQ * BKV / 32;   // S and O elements per thread

// NATTEN window start: centred, shifted inward at the borders (k already clamped to L)
inline int wstart(int i, int k, int L) { return clamp(i - k / 2, 0, L - k); }

// q/k/v: [B, T, H, W, NH, D] sharing strides S = {b, t, h, w}, head stride D; o: contiguous.
// P = {T, H, W, NH, kt, kh, kw}
template <typename T>
inline void na3d(const device T* Q, const device T* K, const device T* V, device T* O, constant long* P,
                 constant long* S, float sl2, uint3 tgid, uint tid, uint sg,
                 threadgroup T* Ks, threadgroup T* Vs, threadgroup uchar4* kc, threadgroup uchar4* win,
                 threadgroup float* rows) {
  const int Tn = P[0], Hn = P[1], Wn = P[2], NH = P[3], kt = P[4], kh = P[5], kw = P[6];
  const int ntw = (Wn + QW - 1) / QW, nth = (Hn + QH - 1) / QH;
  // x enumerates (brick, head) so the heads of one brick, which share cache lines, run together
  const int head = int(tgid.x) % NH, brick = int(tgid.x) / NH;
  const int t0 = brick / (ntw * nth) * QT, h0 = brick / ntw % nth * QH, w0 = brick % ntw * QW;
  const int t1 = min(t0 + QT, Tn), h1 = min(h0 + QH, Hn), w1 = min(w0 + QW, Wn);
  // per-axis union of the brick's windows (starts are monotonic in the query index)
  const int lt = wstart(t0, kt, Tn), lh = wstart(h0, kh, Hn), lw = wstart(w0, kw, Wn);
  const int ut = wstart(t1 - 1, kt, Tn) + kt - lt, uh = wstart(h1 - 1, kh, Hn) + kh - lh;
  const int uw = wstart(w1 - 1, kw, Wn) + kw - lw, uhw = uh * uw, count = ut * uhw;
  const long base = long(tgid.z) * S[0] + head * D;

  // stage the brick's queries (rows outside the volume as zeros) through the K tile
  {
    const int row = tid / 2, c = tid % 2 * 32, r = row % SGQ;
    const int qt = t0 + row / SGQ, qh = h0 + r / QW, qw = w0 + r % QW;
    threadgroup vec<T, 4>* dst = (threadgroup vec<T, 4>*)(Ks + row * D + c);
    if (qt < t1 && qh < h1 && qw < w1) {
      const device vec<T, 4>* src = (const device vec<T, 4>*)(Q + base + qt * S[1] + qh * S[2] + qw * S[3] + c);
      for (int j = 0; j < 8; ++j) dst[j] = src[j];
    } else {
      for (int j = 0; j < 8; ++j) dst[j] = vec<T, 4>(0);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto dqk = matmul2d_descriptor(SGQ, BKV, D, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<dqk, execution_simdgroup> qk;
  constexpr auto dpv = matmul2d_descriptor(SGQ, D, BKV, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroup> pv;
  auto tK = tensor(Ks, dextents<int, 2>{D, BKV}, array<int, 2>{1, D});
  auto tV = tensor(Vs, dextents<int, 2>{D, BKV}, array<int, 2>{1, D});
  // each simdgroup's P reuses its quarter of the K tile once every simdgroup's QK is done
  threadgroup half* Ps = (threadgroup half*)Ks + sg * SGQ * BKV;
  auto tP = tensor(Ps, dextents<int, 2>{BKV, SGQ}, array<int, 2>{1, BKV});
  using KT = __tensor_ops_detail::__remove_addrspace_t<decltype(tK)>;
  using PT = __tensor_ops_detail::__remove_addrspace_t<decltype(tP)>;
  using VT = __tensor_ops_detail::__remove_addrspace_t<decltype(tV)>;
  auto Qc = qk.get_left_input_cooperative_tensor<T, T, float>();
  Qc.load(tensor(Ks + sg * SGQ * D, dextents<int, 2>{D, SGQ}, array<int, 2>{1, D}));
  auto Sc = qk.get_destination_cooperative_tensor<decltype(Qc), KT, float>();
  // per-row block max/sum, running max, running sum, rescale: row-reduction layout, in registers
  auto Rc = qk.get_row_reduction_destination_cooperative_tensor<decltype(Qc), KT, float>();
  auto Mc = qk.get_row_reduction_destination_cooperative_tensor<decltype(Qc), KT, float>();
  auto Lc = qk.get_row_reduction_destination_cooperative_tensor<decltype(Qc), KT, float>();
  auto Ac = qk.get_row_reduction_destination_cooperative_tensor<decltype(Qc), KT, float>();
  for (uint16_t j = 0; j < Mc.get_capacity(); ++j) if (Mc.is_valid_element(j)) { Mc[j] = -FLT_MAX; Lc[j] = 0.0f; }
  auto Oc = pv.get_destination_cooperative_tensor<PT, VT, float>();
  for (uint16_t i = 0; i < Oc.get_capacity(); ++i) if (Oc.is_valid_element(i)) Oc[i] = 0.0f;
  threadgroup float* rv = rows + sg * SGQ;
  auto tR = tensor(rv, dextents<int, 1>{SGQ});
  // the element -> (column, row) maps are fixed for the kernel, so look them up once
  ushort scol[CAP], srow[CAP], orow[CAP];
  for (uint16_t i = 0; i < CAP; ++i) {
    auto x = Sc.get_multidimensional_index(i);
    scol[i] = x[0];
    srow[i] = x[1];
    orow[i] = Oc.get_multidimensional_index(i)[1];
  }

  // this simdgroup's frame; each row's window start (union-relative) as bytes, so a key at
  // union coords c is visible iff all(c - start < kernel) in wrapping uchar math
  const int qt = t0 + int(sg), wt = wstart(min(qt, t1 - 1), kt, Tn) - lt;
  const bool active = qt < t1;
  threadgroup uchar4* rwin = win + sg * SGQ;
  if (tid % 32 < SGQ) {
    const int r = tid % 32;
    rwin[r] = uchar4(wt, wstart(min(h0 + r / QW, h1 - 1), kh, Hn) - lh, wstart(min(w0 + r % QW, w1 - 1), kw, Wn) - lw, 0);
  }
  const uchar4 ksz = uchar4(kt, kh, kw, 1);

  for (int kb = 0; kb < count; kb += BKV) {
    threadgroup_barrier(mem_flags::mem_threadgroup);   // previous PV is done with K (as P) and V
    {
      const int j = tid / 2, c = tid % 2 * 32, f = kb + j;
      threadgroup vec<T, 4>* kd = (threadgroup vec<T, 4>*)(Ks + j * D + c);
      threadgroup vec<T, 4>* vd = (threadgroup vec<T, 4>*)(Vs + j * D + c);
      if (f < count) {
        const int ft = f / uhw, fh = f / uw % uh, fw = f % uw;
        const long off = base + (lt + ft) * S[1] + (lh + fh) * S[2] + (lw + fw) * S[3] + c;
        const device vec<T, 4>* ks = (const device vec<T, 4>*)(K + off);
        const device vec<T, 4>* vs = (const device vec<T, 4>*)(V + off);
        for (int i = 0; i < 8; ++i) { kd[i] = ks[i]; vd[i] = vs[i]; }
        if (c == 0) kc[j] = uchar4(ft, fh, fw, 0);
      } else {
        for (int i = 0; i < 8; ++i) { kd[i] = vec<T, 4>(0); vd[i] = vec<T, 4>(0); }
        if (c == 0) kc[j] = uchar4(255, 255, 255, 255);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // skip blocks whose frames are all outside this simdgroup's window
    const bool run = active && kb / uhw < wt + kt && (min(kb + BKV, count) - 1) / uhw >= wt;
    if (run) {
      qk.run(Qc, tK, Sc);
      for (uint16_t i = 0; i < CAP; ++i)
        Sc[i] = all(kc[scol[i]] - rwin[srow[i]] < ksz) ? Sc[i] * sl2 : -FLT_MAX;
      // keep a row's stale max unless the block's beats it by 2^8 (p <= 256 fits P; O and l share the offset)
      reduce_rows(Sc, Rc, reduction_operation::max, -FLT_MAX);
      bool moved = false;
      for (uint16_t j = 0; j < Rc.get_capacity(); ++j) {
        if (!Rc.is_valid_element(j)) continue;
        const bool mv = Rc[j] > Mc[j] + 8.0f;
        Ac[j] = mv ? fast::exp2(Mc[j] - Rc[j]) : 1.0f;
        Mc[j] = mv ? Rc[j] : Mc[j];
        moved |= mv;
      }
      moved = simd_any(moved);
      for (uint16_t i = 0; i < CAP; ++i) {
        const float s = Sc[i];
        Sc[i] = s > -FLT_MAX ? fast::exp2(s - *Mc.map_iterator(Sc.get_iterator(i))) : 0.0f;
      }
      reduce_rows(Sc, Rc, reduction_operation::sum, 0.0f);
      for (uint16_t j = 0; j < Rc.get_capacity(); ++j) if (Rc.is_valid_element(j)) Lc[j] = Lc[j] * Ac[j] + Rc[j];
      if (moved) {
        Ac.store(tR);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < CAP; ++i) Oc[i] *= rv[orow[i]];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // every simdgroup is done reading K
    if (run) {
      for (uint16_t i = 0; i < CAP; ++i) Ps[srow[i] * BKV + scol[i]] = half(Sc[i]);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      pv.run(tP, tV, Oc);
    }
  }
  if (!active) return;
  for (uint16_t j = 0; j < Lc.get_capacity(); ++j) if (Lc.is_valid_element(j)) Lc[j] = 1.0f / Lc[j];
  simdgroup_barrier(mem_flags::mem_threadgroup);
  Lc.store(tR);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint16_t i = 0; i < Oc.get_capacity(); ++i) {
    if (!Oc.is_valid_element(i)) continue;
    auto idx = Oc.get_multidimensional_index(i);
    const int r = idx[1], qh = h0 + r / QW, qw = w0 + r % QW;
    if (qh < h1 && qw < w1)
      O[(((ulong(tgid.z) * Tn + qt) * Hn + qh) * Wn + qw) * NH * D + head * D + idx[0]] = T(Oc[i] * rv[r]);
  }
}

#define NA3D(NAME, T) \
kernel void NAME(const device T* Q [[buffer(0)]], const device T* K [[buffer(1)]], const device T* V [[buffer(2)]], \
                 device T* O [[buffer(3)]], constant long* P [[buffer(4)]], constant long* S [[buffer(5)]], \
                 constant float& sl2 [[buffer(6)]], uint3 tgid [[threadgroup_position_in_grid]], \
                 uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
  threadgroup T Ks[BKV * D]; \
  threadgroup T Vs[BKV * D]; \
  threadgroup uchar4 kc[BKV]; \
  threadgroup uchar4 win[QT * SGQ]; \
  threadgroup float rows[QT * SGQ]; \
  na3d<T>(Q, K, V, O, P, S, sl2, tgid, tid, sg, Ks, Vs, kc, win, rows); \
}
NA3D(na3d_bf16, bfloat)
NA3D(na3d_f16, half)
"""

_SUFFIX = {torch.bfloat16: "bf16", torch.float16: "f16"}
_LOG2E = 1.4426950408889634


def _launch(q, k, v, kernels, scale):
    B, T, H, W, NH, D = q.shape
    o = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    bricks = -(-T // _BRICK) * -(-H // _BRICK) * -(-W // _BRICK)
    getattr(_lib, "na3d_" + _SUFFIX[q.dtype])(
        q, k, v, o, [T, H, W, NH, *kernels], list(q.stride()[:4]), float(scale * _LOG2E),
        threads=(bricks * NH * 128, 1, B), group_size=(128, 1, 1))
    return o


def _clamped_kernels(kernel_size, dims):
    """Per-axis windows clamped to their axis, or None if unsupported or too wide for uchar."""
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * 3
    try:
        if len(kernel_size) != 3 or not all(type(x) is int and x >= 1 for x in kernel_size):
            return None
    except TypeError:
        return None
    ks = [min(x, d) for x, d in zip(kernel_size, dims)]
    # a brick's union window along an axis is at most window + 3, and at most the axis
    if any(x > _MAX_EXT or min(d, x + _BRICK - 1) > _MAX_EXT for x, d in zip(ks, dims)):
        return None
    return ks


def _plan(q, k, v, kernel_size, is_causal, scale):
    """Clamped kernels if the fused kernel can serve this call, else None."""
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
        return None
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return None
    if isinstance(is_causal, bool):
        is_causal = [is_causal] * 3
    if is_causal is not None and any(is_causal):
        return None
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float))):
        return None
    if not (q.device.type == "mps" and k.device == q.device and v.device == q.device
            and q.dtype in _SUFFIX and k.dtype == q.dtype and v.dtype == q.dtype
            and q.dim() == 6 and q.shape[-1] == 64 and k.shape == q.shape and v.shape == q.shape
            and q.numel() > 0 and q.stride() == k.stride() == v.stride()
            and q.stride(-1) == 1 and q.stride(-2) == 64):
        return None
    # the shader loads 4-element vectors
    if any(s % 4 for s in q.stride()[:4]) or any(t.storage_offset() % 4 for t in (q, k, v)):
        return None
    return _clamped_kernels(kernel_size, q.shape[1:4])


def _na3d(q, k, v, kernel_size, is_causal=None, scale=None):
    kernels = _plan(q, k, v, kernel_size, is_causal, scale)
    # the self-check (shader build) runs only for a call the kernel could serve
    if kernels is not None and _caps.kernel_ready("na3d", _verify) and q.dtype in _good:
        try:
            return _launch(q, k, v, kernels, q.shape[-1] ** -0.5 if scale is None else scale)
        except Exception as e:
            # latch off: verification passed, so a dispatch failure recurs on every later call
            _caps.mark_kernel_failed("na3d")
            print(f"{TAG} kernel failed ({e!r}); using comfy_kitchen's na3d for the rest of this session.")
    return _orig(q, k, v, kernel_size, is_causal, scale)


# ragged bricks on every axis, a kernel clamped to its axis, and one equal to it;
# the second case reads q/k/v as views of one fused qkv buffer
_CASES = (((1, 5, 9, 11, 2, 64), (3, 5, 5), False), ((2, 3, 6, 13, 1, 64), (5, 3, 7), True))


def _verify():
    global _lib
    try:
        _lib = torch.mps.compile_shader(_MSL)
    except Exception as e:
        print(f"{TAG} shader did not compile; using comfy_kitchen's na3d: {e!r}")
        return False
    from comfy_kitchen.backends.eager.na import na3d as ref_na3d
    _good.clear()
    g = torch.Generator().manual_seed(0)
    for dtype in _SUFFIX:
        errs = []
        for shape, ks, views in _CASES:
            qkv = torch.randn(*shape[:-2], 3, *shape[-2:], generator=g).to(dtype)
            ref = ref_na3d(*(t.float() for t in qkv.unbind(-3)), list(ks), None, 0.125)
            q, k, v = qkv.to("mps").unbind(-3)
            if not views:
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            out = _launch(q, k, v, [min(a, d) for a, d in zip(ks, shape[1:4])], 0.125)
            errs.append(((out.cpu().float() - ref).norm() / ref.norm()).item())
        if max(errs) < 0.01:
            _good.add(dtype)
        else:
            print(f"{TAG} {dtype} self-check mismatch (rel errs {[f'{e:.3g}' for e in errs]}); "
                  f"that dtype uses comfy_kitchen's na3d.")
    return bool(_good)


def install():
    global _installed, _orig
    if _installed or sys.platform != "darwin" or not _caps.is_mps():
        return
    if not _caps.resolve("ASFP8_NA3D", default_on=True, cap=_caps.shader_gate):
        return
    try:
        import comfy_kitchen
        orig = comfy_kitchen.na3d
    except Exception:
        return
    if getattr(orig, "_asfp8_na3d", False):
        _installed = True
        return
    _orig = orig
    _na3d._asfp8_na3d = True
    comfy_kitchen.na3d = _na3d
    _installed = True
    print(f"{TAG} comfy_kitchen.na3d on MPS -> fused matmul2d neighborhood attention (self-check on first use).")
