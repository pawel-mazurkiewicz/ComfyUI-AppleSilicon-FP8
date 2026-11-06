"""Count PyTorch's reusable MPS allocator pool as free memory.

Stock ComfyUI reports psutil "available" for MPS, which cannot see memory this process
already holds in the allocator pool, so it evicts models it could keep. Disable with
ASFP8_FREE_MEM=off.
"""

import os
import sys

import torch

TAG = "[AppleSilicon-FP8/free-mem]"
_OFF = ("off", "0", "false", "none", "no")


def _enabled():
    return os.environ.get("ASFP8_FREE_MEM", "").strip().lower() not in _OFF


def _mlx_held():
    # MLX buffers share the Metal device total; they are not reusable by torch.
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return 0
    try:
        return mx.get_active_memory() + mx.get_cache_memory()
    except Exception:
        try:
            return mx.metal.get_active_memory() + mx.metal.get_cache_memory()
        except Exception:
            raise  # fail closed: unreadable MLX counters must not inflate the pool


def _reusable_pool():
    held = torch.mps.driver_allocated_memory() - torch.mps.current_allocated_memory()
    return max(0, held - _mlx_held())


_failed = False


def _latch_off(exc):
    global _failed
    _failed = True
    print(f"{TAG} disabled for this session after {type(exc).__name__}: {exc}", flush=True)


def _supported():
    return sys.platform == "darwin" and torch.backends.mps.is_available()


def install():
    if not _supported():
        return False
    try:
        import comfy.model_management as mm
    except ImportError:
        return False

    if getattr(mm.get_free_memory, "_asfp8_wrapped", False):
        return True
    if not _enabled():
        return False
    original = mm.get_free_memory

    def get_free_memory(dev=None, torch_free_too=False):
        d = mm.get_torch_device() if dev is None else dev
        if getattr(d, "type", None) != "mps" or _failed or not _enabled():
            return original(dev, torch_free_too)
        try:
            import comfy.system_memory as sm

            avail = sm.virtual_memory_available()   # first: a later pool release only under-counts
            pool = _reusable_pool()
            total = avail + pool
        except Exception as e:
            _latch_off(e)
            return original(dev, torch_free_too)
        return (total, pool) if torch_free_too else total

    get_free_memory._asfp8_wrapped = True
    mm.get_free_memory = get_free_memory
    print(f"{TAG} counting the reusable MPS pool as free memory", flush=True)
    return True
