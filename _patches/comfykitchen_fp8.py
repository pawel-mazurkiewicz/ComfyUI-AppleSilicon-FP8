"""Fix: comfy_kitchen FP8 quantization on MPS (e.g. Ideogram 4).

The eager backend dequantizes with plain fp8 casts, which MPS rejects; replace them
with the LUT decode, and reroute the NVFP4/MXFP8 block-scale swizzles through the CPU.
NVFP4 dequant instead runs as one compile_shader pass once it has reproduced the eager
result exactly on this machine; until then, or if it ever fails, it takes the CPU path.
The registry resolves implementations by getattr() at call time, so overwriting the
module attributes is picked up by every dispatch.
"""

import sys

import torch

from . import _caps
from ._common import FP8_DTYPES, decode_fp8

TAG = "[AppleSilicon-FP8/comfy_kitchen]"

_installed = False


def _cpu_dequant_on_mps(orig):
    """Run a comfy_kitchen eager dequant on CPU when its inputs are on MPS, then move
    the float result back. The block-scales are tiny, so the round-trip is cheap."""
    def wrapped(*args, **kwargs):
        dev = None
        for a in (*args, *kwargs.values()):
            if isinstance(a, torch.Tensor):
                dev = a.device
                break
        if dev is None or dev.type != "mps":
            return orig(*args, **kwargs)
        cargs = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        ckwargs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
        return orig(*cargs, **ckwargs).to(dev)

    return wrapped


# NVFP4 dequant shader ported from Illuminus7/ComfyUI-AppleSilicon-int8-Fast (MIT)
_NVFP4_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant float E2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline float e4m3(uchar u) {
  const uint e = (u >> 3) & 15, m = u & 7;
  const float v = e ? ldexp(1.0f + float(m) / 8.0f, int(e) - 7) : ldexp(float(m) / 8.0f, -6);
  return (u & 128) ? -v : v;
}

// one thread per packed byte -> 2 outputs; block scales in cuBLAS' swizzled 128x4 tiles.
// Rounding follows comfy_kitchen: scale = T(T(tensor_scale) * T(block_scale)), out = T(fp4 * scale)
template <typename T>
inline void dequant(const device uchar* q, const device float* ts, const device uchar* bs, device T* out,
                    uint C, uint n_cb, uint hi_first, uint i) {
  const uint r = i / (C / 2), c = (i % (C / 2)) * 2;
  const uint blk = c / 16, rb = r / 128, rr = r % 128, cb = blk / 4;
  const uchar sb = bs[(rb * n_cb + cb) * 512 + (rr % 32) * 16 + (rr / 32) * 4 + blk % 4];
  const T scale = T(float(T(ts[0])) * float(T(e4m3(sb))));
  const uchar b = q[i];
  const uchar first = hi_first ? (b >> 4) : (b & 15), second = hi_first ? (b & 15) : (b >> 4);
  // E4M3 NaN (0x7F/0xFF) poisons the whole block, zeros included
  const bool nan_sb = (sb & 127) == 127;
  const T nan_v = T(as_type<float>(0x7FC00000u));
  out[ulong(r) * C + c] = nan_sb ? nan_v : T(E2M1[first] * float(scale));
  out[ulong(r) * C + c + 1] = nan_sb ? nan_v : T(E2M1[second] * float(scale));
}

#define DQ(NAME, T) \
kernel void NAME(const device uchar* q [[buffer(0)]], const device float* ts [[buffer(1)]], const device uchar* bs [[buffer(2)]], \
                 device T* out [[buffer(3)]], constant uint& C [[buffer(4)]], constant uint& n_cb [[buffer(5)]], \
                 constant uint& hi_first [[buffer(6)]], uint i [[thread_position_in_grid]]) { \
  dequant<T>(q, ts, bs, out, C, n_cb, hi_first, i); \
}
DQ(nvfp4_bf16, bfloat)
DQ(nvfp4_f16, half)
DQ(nvfp4_f32, float)
"""

_NVFP4_KERNEL = "nvfp4_dequant"
_NVFP4_SUFFIX = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
_nvfp4_lib = None


def _nvfp4_mps(lib, qx, per_tensor_scale, block_scales, output_type, hi_first):
    R, C = qx.shape[0], qx.shape[1] * 2
    out = torch.empty(R, C, device=qx.device, dtype=output_type)
    n_cb = (C // 16 + 3) // 4
    getattr(lib, "nvfp4_" + _NVFP4_SUFFIX[output_type])(
        qx.contiguous(), per_tensor_scale.reshape(1).to(qx.device), block_scales.view(torch.uint8), out,
        C, n_cb, int(bool(hi_first)), threads=(qx.numel(), 1, 1))
    return out


def _nvfp4_shader_args(qx, per_tensor_scale, block_scales, output_type=torch.bfloat16, hi_first=True):
    """The call as the shader takes it, or None for anything outside what it is verified on."""
    if qx.device.type != "mps" or block_scales.device.type != "mps":
        return None
    if qx.dim() != 2 or qx.dtype != torch.uint8 or output_type not in _NVFP4_SUFFIX:
        return None
    if per_tensor_scale.dtype != torch.float32 or per_tensor_scale.numel() != 1:
        return None
    if block_scales.dtype != torch.float8_e4m3fn or not block_scales.is_contiguous():
        return None
    R, C = qx.shape[0], qx.shape[1] * 2
    if not (0 < qx.numel() < 2 ** 31 and C % 16 == 0):  # the thread index is 32-bit
        return None
    # eager raises on a mis-sized swizzle; the shader would read past the buffer
    if block_scales.numel() != ((R + 127) // 128) * ((C // 16 + 3) // 4) * 512:
        return None
    return qx, per_tensor_scale, block_scales, output_type, hi_first


def _same_bits(a, b):
    """Equal bit for bit (signed zeros included), NaNs matching by position only."""
    na = a.isnan()
    if not torch.equal(na, b.isnan()):
        return False
    bits = torch.int32 if a.dtype == torch.float32 else torch.int16
    return torch.equal(a.view(bits)[~na], b.view(bits)[~na])


def _nvfp4_verify(orig):
    """kernel_ready() check: compile, then match the eager result exactly on a ragged tensor."""
    def verify():
        global _nvfp4_lib
        try:
            lib = torch.mps.compile_shader(_NVFP4_MSL)
        except Exception as e:
            print(f"{TAG} NVFP4 dequant shader did not compile; using the CPU path: {e!r}")
            return False
        from comfy_kitchen.float_utils import to_blocked
        g = torch.Generator().manual_seed(0)
        R, nb = 200, 37  # ragged against the 128-row and 4-block tiles
        qx = torch.randint(0, 256, (R, nb * 8), generator=g, dtype=torch.uint8)
        # every E4M3 byte appears as a block scale
        sb = to_blocked((torch.arange(R * nb) % 256).to(torch.uint8).view(R, nb), flatten=False)
        bs = sb.view(torch.float8_e4m3fn)
        ts = torch.tensor(0.37)
        for dt in _NVFP4_SUFFIX:
            for hi in (True, False):
                ref = orig(qx, ts, bs, dt, hi)
                got = _nvfp4_mps(lib, qx.to("mps"), ts.to("mps"), bs.to("mps"), dt, hi).cpu()
                if not _same_bits(got, ref):
                    print(f"{TAG} NVFP4 dequant shader differs from the eager result "
                          f"({dt}, hi_first={hi}); using the CPU path.")
                    return False
        _nvfp4_lib = lib
        return True
    return verify


def _nvfp4_dequant(orig, cpu_path):
    """dequantize_nvfp4 on MPS through the shader; anything else keeps the CPU round-trip."""
    verify = _nvfp4_verify(orig)

    def dequantize_nvfp4(*args, **kwargs):
        try:
            call = _nvfp4_shader_args(*args, **kwargs)
        except Exception:
            call = None  # not a call we recognise; the old path decides what to do with it
        if call is not None and _caps.kernel_ready(_NVFP4_KERNEL, verify):
            try:
                return _nvfp4_mps(_nvfp4_lib, *call)
            except Exception as e:
                # verified, so this repeats on every later call
                print(f"{TAG} NVFP4 dequant shader failed ({e!r}); using the CPU path for this session.")
                _caps.mark_kernel_failed(_NVFP4_KERNEL)
        return cpu_path(*args, **kwargs)

    return dequantize_nvfp4


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    try:
        import comfy_kitchen  # noqa: F401  (ensures backends register)
        from comfy_kitchen.registry import registry
        import comfy_kitchen.backends.eager.quantization as qmod
    except Exception:
        return  # comfy_kitchen not installed; nothing to patch

    # separate from the gate above so a build without it still gets the fp8 fix
    try:
        import comfy_kitchen.float_utils as fumod
    except ImportError:
        fumod = None

    eager = registry._backends.get("eager")
    if eager is None:
        return

    orig_dequant = eager.dequantize_per_tensor_fp8
    orig_quant = eager.quantize_per_tensor_fp8

    def dequantize_per_tensor_fp8(x, scale, output_type=torch.bfloat16):
        if x.device.type == "mps" and x.dtype in FP8_DTYPES:
            return decode_fp8(x).to(output_type) * scale.to(output_type)
        return orig_dequant(x, scale, output_type)

    def quantize_per_tensor_fp8(x, scale, output_type=torch.float8_e4m3fn):
        if x.device.type == "mps" and output_type in FP8_DTYPES:
            lp_max = (
                qmod.F8_E4M3_MAX if output_type == torch.float8_e4m3fn else qmod.F8_E5M2_MAX
            )
            temp = torch.clamp(x * (1.0 / scale).to(x.dtype), -lp_max, lp_max)
            return temp.to("cpu").to(output_type).to(x.device)  # FP8 cast unsupported on MPS
        return orig_quant(x, scale, output_type)

    for mod in (eager, qmod):
        mod.dequantize_per_tensor_fp8 = dequantize_per_tensor_fp8
        mod.quantize_per_tensor_fp8 = quantize_per_tensor_fp8

    # getattr-guarded: a no-op on comfy_kitchen builds predating these formats
    nvfp4_mxfp8 = []
    shader = False
    for fname in ("dequantize_nvfp4", "dequantize_mxfp8"):
        orig_fn = getattr(eager, fname, None)
        if orig_fn is None:
            continue
        wrapped = _cpu_dequant_on_mps(orig_fn)
        if fname == "dequantize_nvfp4" and _caps.has_compile_shader():
            wrapped = _nvfp4_dequant(orig_fn, wrapped)
            shader = True
        for mod in (eager, qmod):
            setattr(mod, fname, wrapped)
        nvfp4_mxfp8.append(fname)

    # to_blocked pads via a strided fp8 copy MPS has no kernel for. The dtype guard
    # leaves MXFP8's uint8 E8M0 scales alone; patch every module resolving the name.
    orig_to_blocked = getattr(fumod, "to_blocked", None) if fumod is not None else None
    if orig_to_blocked is not None:
        def to_blocked(input_matrix, *args, **kwargs):
            if input_matrix.device.type == "mps" and input_matrix.dtype in FP8_DTYPES:
                return orig_to_blocked(input_matrix.cpu(), *args, **kwargs).to(input_matrix.device)
            return orig_to_blocked(input_matrix, *args, **kwargs)

        for mod in (fumod, qmod, comfy_kitchen):
            if getattr(mod, "to_blocked", None) is not None:
                mod.to_blocked = to_blocked
        nvfp4_mxfp8.append("to_blocked")

    _installed = True
    note = "; dequantize_nvfp4 in a shader once it checks out" if shader else ""
    extra = f" (+ {', '.join(nvfp4_mxfp8)} via CPU{note})" if nvfp4_mxfp8 else ""
    print(f"{TAG} patched comfy_kitchen eager FP8 dequantize/quantize for MPS{extra}.")
