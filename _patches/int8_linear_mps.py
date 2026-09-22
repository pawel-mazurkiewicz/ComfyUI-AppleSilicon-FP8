"""Fix: INT8 models crawl on MPS even after the _int_mm GPU patch.

The int8-fast node's wide-batch path quantizes activations and matmuls with _int_mm,
which patch #12 can only keep on the GPU in float32. Route both wide-batch entry points
through the node's own small-batch path instead — dequantize the weight once, then a
native bf16 GEMM, which is also weight-only int8 and so matches a plain bf16 forward.
int8-fast usually imports after us, hence the one-shot post-import hook.
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
    if not _scan_loaded():   # it may already be imported, or not yet
        sys.meta_path.insert(0, _Finder())
