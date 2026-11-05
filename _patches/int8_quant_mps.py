"""Single-pass per-row INT8 activation quant on MPS, bit-identical to comfy_kitchen's
quantize_int8_rowwise. ASFP8_INT8_NATIVE_QUANT=off routes everything to kitchen.
"""
from __future__ import annotations

import torch

from . import _caps

TAG = "[AppleSilicon-FP8/int8_quant]"
NAME = "int8_quant"

_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TG    = 256;
constant constexpr uint NSIMD = TG / 32;

// float -> bf16 / fp16 value, round-to-nearest-even, returned as an exact float
inline float rne_t(float f, bfloat) {
    uint u = as_type<uint>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return f;
    u += 0x7fffu + ((u >> 16) & 1u);
    return as_type<float>(u & 0xffff0000u);
}

inline float rne_t(float f, half) {
    const uint u = as_type<uint>(f);
    const uint s = u & 0x80000000u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return f;
    if (a >= 0x477ff000u) return as_type<float>(s | 0x7f800000u);
    if (a >= 0x38800000u) return as_type<float>(s | ((a + 0xfffu + ((a >> 13) & 1u)) & 0xffffe000u));
    if (a < 0x33000000u) return as_type<float>(s);
    const uint m = (a & 0x7fffffu) | 0x800000u, sh = 126u - (a >> 23);
    const uint rem = m & ((1u << sh) - 1u), hw = 1u << (sh - 1u);
    uint q = m >> sh;
    q += (rem > hw || (rem == hw && (q & 1u))) ? 1u : 0u;
    return as_type<float>(s | as_type<uint>(float(q) * as_type<float>(0x33800000u)));
}

inline float tiny_t(bfloat) { return as_type<float>(0x00800000u); }
inline float tiny_t(half)   { return as_type<float>(0x38800000u); }

kernel void quant_rowwise(
    device const @T@* x     [[buffer(0)]],
    device       char* q    [[buffer(1)]],
    device       float* scl [[buffer(2)]],
    constant     uint& K    [[buffer(3)]],
    constant     uint& ny   [[buffer(4)]],
    constant     uint& rows [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    uint  sgid  [[simdgroup_index_in_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]])
{
    const uint row = tgpig.z * ny + tgpig.y;
    if (row >= rows) return;
    const ulong base = ulong(row) * ulong(K);

    float m = 0.0f;
    for (uint k = lid; k < K; k += TG) m = max(m, fabs(float(x[base + k])));
    m = simd_max(m);
    threadgroup float part[NSIMD];
    if (lane == 0) part[sgid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = simd_max(lane < NSIMD ? part[lane] : 0.0f);
        if (lane == 0) part[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float s32 = max(precise::divide(part[0], 127.0f), 1e-30f);
    if (lid == 0) scl[row] = s32;
    float st = rne_t(s32, @T@(0));
    if (st == 0.0f) st = tiny_t(@T@(0));

    for (uint k = lid; k < K; k += TG) {
        const float v = rne_t(precise::divide(float(x[base + k]), st), @T@(0));
        q[base + k] = char(clamp(rint(v), -128.0f, 127.0f));
    }
}
"""

_MSL_T = {torch.bfloat16: "bfloat", torch.float16: "half"}
_TG = 256
_NY = 32768
_libs: dict = {}


def _kitchen(x2d):
    from comfy_kitchen.backends.eager import quantization
    return quantization.quantize_int8_rowwise(x2d)


def _lib(dtype):
    lib = _libs.get(dtype)
    if lib is None:
        lib = torch.mps.compile_shader(_MSL.replace("@T@", _MSL_T[dtype]))
        _libs[dtype] = lib
    return lib


def _native(x2d):
    x2d = x2d.contiguous()
    M, K = x2d.shape
    q = torch.empty((M, K), dtype=torch.int8, device=x2d.device)
    s = torch.empty((M, 1), dtype=torch.float32, device=x2d.device)
    ny = min(M, _NY)
    nz = (M + ny - 1) // ny
    _lib(x2d.dtype).quant_rowwise(x2d, q, s, K, ny, M,
                                  threads=(_TG, ny, nz), group_size=(_TG, 1, 1))
    return q, s


def _adversarial(dtype):
    K = 1000
    k = torch.arange(K, dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    rows = torch.randn(10, K, generator=g) * 3
    rows[0] = 0
    rows[1] = torch.where(k % 2 == 0, 1.0, -1.0) * ((k % 127) + 0.5) * 2.0 ** -6
    rows[1, 0] = 127 * 2.0 ** -6
    rows[2] = torch.finfo(dtype).max
    rows[3] *= 1e-39 if dtype == torch.bfloat16 else 1e-5
    rows[4] = 0
    rows[4, 7] = -2.75
    return rows.to(dtype).to("mps")


def _verify():
    for dtype in _MSL_T:
        x = _adversarial(dtype)
        q, s = _native(x)
        qr, sr = _kitchen(x)
        if not (torch.equal(q, qr) and torch.equal(s, sr)):
            print(f"{TAG} {dtype} self-check mismatch; using comfy_kitchen quant.", flush=True)
            return False
    return True


def verify():
    return _caps.has_compile_shader() and _caps.kernel_ready(NAME, _verify)


def quantize_rowwise(x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (x2d.device.type == "mps" and x2d.dtype in _MSL_T and x2d.dim() == 2
            and x2d.numel() > 0
            and _caps.resolve("ASFP8_INT8_NATIVE_QUANT", True, _caps.has_compile_shader)
            and verify()):
        try:
            return _native(x2d)
        except Exception as e:
            _caps.mark_kernel_failed(NAME)
            print(f"{TAG} kernel failed ({e!r}); using comfy_kitchen quant from here on.",
                  flush=True)
    return _kitchen(x2d)
