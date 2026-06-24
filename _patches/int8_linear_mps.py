"""Fix: INT8 models crawl on MPS (~144 s/step) even after the _int_mm GPU patch.

The `int8-fast` custom node's quantized Linear has two forward paths (int8_quant.py):

    if x_2d.shape[0] > 16:                       # the normal case: many tokens
        y = int8_forward_dynamic[_per_row](...)  # quantizes x to int8, then torch._int_mm
    else:                                        # "small batch fallback"
        w = dequantize(weight, w_scale).to(dtype) # int8 weight -> compute dtype
        y = F.linear(x, w, bias)                  # native GEMM

The wide-batch path quantizes activations to int8 and matmuls with `torch._int_mm`.
On MPS `_int_mm` has no Metal kernel; patch #12 keeps it on the GPU but only in
float32, which is 3-5x slower than bf16 and doubles the working set (fp32 temps on
top of a 12 GB model -> memory pressure). Image diffusion runs thousands of tokens,
so it's always on this slow path.

The fix is the node's *own* small-batch path: dequantize the int8 weight to the
bf16 compute dtype once and call `F.linear`, which uses MPS's native, double-buffered
bf16 GEMM — the fastest matmul available on Apple Silicon (a custom int8 matmul2d
kernel only beats it if fully threadgroup-staged; simple-tiled loses to MPS's own
`a @ b` at every size). Measured 3.5-4.7x over the float32 _int_mm path on FLUX-shaped
Linears, and accuracy is equal-or-better: it skips int8 activation quantization
(weight-only int8), so it matches a plain bf16 forward by construction.

So on MPS we route `int8_forward_dynamic` / `int8_forward_dynamic_per_row` through
the dequantize+F.linear path for every batch size. Non-MPS keeps the original
(Triton on CUDA, int8 `_int_mm` elsewhere).

int8-fast usually imports *after* this node (custom nodes load alphabetically), so
we patch its `int8_quant` module via a one-shot post-import hook. No-op if int8-fast
isn't installed.
"""

import sys

import torch

TAG = "[AppleSilicon-FP8/int8_linear]"

_installed = False   # our install() ran
_patched = False     # int8-fast's module actually patched
_orig_dyn = None
_orig_dyn_pr = None
_dequantize = None


def _mps_linear(x, weight, weight_scale, bias, compute_dtype):
    # Mirror int8-fast's own small-batch fallback: int8 weight -> bf16, native GEMM.
    w = _dequantize(weight, weight_scale).to(compute_dtype)
    b = bias.to(compute_dtype) if bias is not None else None
    return torch.nn.functional.linear(x, w, b)


def _dyn(x, weight, weight_scale, bias, compute_dtype):
    if x.device.type == "mps":
        return _mps_linear(x, weight, weight_scale, bias, compute_dtype)
    return _orig_dyn(x, weight, weight_scale, bias, compute_dtype)


def _dyn_pr(x, weight, weight_scale, bias, compute_dtype):
    if x.device.type == "mps":
        return _mps_linear(x, weight, weight_scale, bias, compute_dtype)
    return _orig_dyn_pr(x, weight, weight_scale, bias, compute_dtype)


def _apply(mod):
    """Patch int8-fast's int8_quant module in place (idempotent)."""
    global _patched, _orig_dyn, _orig_dyn_pr, _dequantize
    if _patched:
        return
    if not (hasattr(mod, "int8_forward_dynamic")
            and hasattr(mod, "int8_forward_dynamic_per_row")
            and hasattr(mod, "dequantize")):
        return
    _orig_dyn = mod.int8_forward_dynamic
    _orig_dyn_pr = mod.int8_forward_dynamic_per_row
    _dequantize = mod.dequantize
    mod.int8_forward_dynamic = _dyn
    mod.int8_forward_dynamic_per_row = _dyn_pr
    _patched = True
    print(f"{TAG} int8-fast wide-batch matmul routed via MPS native bf16 GEMM (was fp32 _int_mm).")


def _scan_loaded():
    for mod in list(sys.modules.values()):
        try:
            name = getattr(mod, "__name__", "")
            if mod is not None and name.rsplit(".", 1)[-1] == "int8_quant" \
                    and hasattr(mod, "int8_forward_dynamic"):
                _apply(mod)
                return _patched
        except Exception:
            continue
    return False


class _Finder:
    """Wraps int8_quant's loader to patch it right after it executes."""

    def find_spec(self, fullname, path, target=None):
        if fullname.rsplit(".", 1)[-1] != "int8_quant":
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig_exec=orig_exec):
            _orig_exec(module)
            try:
                _apply(module)
            except Exception as e:  # never break the import
                print(f"{TAG} post-import patch failed: {e}")

        try:
            spec.loader.exec_module = exec_module
        except Exception:
            return None
        return spec


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    _installed = True
    # int8-fast may already be imported (loads before us) or not yet (loads after).
    if not _scan_loaded():
        sys.meta_path.insert(0, _Finder())
