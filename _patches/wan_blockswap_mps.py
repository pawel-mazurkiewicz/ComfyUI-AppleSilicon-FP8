"""Switch: neutralize ComfyUI-WanVideoWrapper block swap on MPS.

Block swap offloads transformer blocks to the CPU and streams them back to the
GPU on demand, to fit big models into scarce NVIDIA VRAM. The swap-in
(`block.to(self.main_device)`) relies on CUDA events to synchronize the async
copy; on MPS that synchronization doesn't hold, so a block's parameters (e.g.
`self.modulation`) are still on the CPU when the block computes:

    RuntimeError: Expected all tensors to be on the same device, but found at
                  least two devices, mps:0 and cpu!   (model.py get_mod: modulation + e)

On Apple Silicon there is no separate VRAM — memory is unified — so block swap
is pure downside: it adds cpu<->mps copies that save nothing and break on MPS.

This wraps WanModel.forward so that, on MPS, every block (and the vace blocks)
is made resident on main_device and the offload flags are cleared before the
real forward runs — i.e. the model behaves as if block swap were never enabled.
This reproduces the known-good "no block swap node" state regardless of what the
downloaded workflow configured.

Disable with ASFP8_NEUTRALIZE_BLOCKSWAP=off.

Because ComfyUI-WanVideoWrapper imports after this plugin, we register a small
post-import hook on sys.meta_path and patch WanModel as soon as its module loads.
"""

import os
import sys
import importlib.abc

import torch

TAG = "[AppleSilicon-FP8/wan_blockswap]"

_TARGET_SUFFIX = "wanvideo.modules.model"
_installed = False


def _enabled():
    return os.environ.get("ASFP8_NEUTRALIZE_BLOCKSWAP", "on").lower() not in (
        "off", "0", "false", "no",
    )


def _patch_wanmodel(module):
    WanModel = getattr(module, "WanModel", None)
    if WanModel is None or getattr(WanModel, "_asfp8_blockswap_patched", False):
        return

    _orig_forward = WanModel.forward

    def _forward(self, *args, **kwargs):
        if _enabled():
            main_dev = getattr(self, "main_device", None)
            is_mps = getattr(main_dev, "type", None) == "mps"
            # Fall back to detecting MPS via the first block's params if needed.
            if main_dev is None:
                is_mps = torch.backends.mps.is_available()
            if is_mps:
                # Clear every offload knob so the real forward never streams.
                self.blocks_to_swap = 0
                if hasattr(self, "vace_blocks_to_swap"):
                    self.vace_blocks_to_swap = 0
                if hasattr(self, "prefetch_blocks"):
                    self.prefetch_blocks = 0
                self.offload_txt_emb = False
                self.offload_img_emb = False
                # Make all blocks resident on the compute device.
                if main_dev is not None:
                    for attr in ("blocks", "vace_blocks"):
                        mods = getattr(self, attr, None)
                        if mods is not None:
                            for blk in mods:
                                blk.to(main_dev)
        return _orig_forward(self, *args, **kwargs)

    WanModel.forward = _forward
    WanModel._asfp8_blockswap_patched = True
    print(f"{TAG} WanModel block swap neutralized on MPS (set ASFP8_NEUTRALIZE_BLOCKSWAP=off to disable).")


class _PostImportHook(importlib.abc.MetaPathFinder):
    """Patch a module right after it finishes importing, without forcing it early."""

    def __init__(self, suffix, callback):
        self._suffix = suffix
        self._callback = callback
        self._busy = False

    def find_spec(self, fullname, path=None, target=None):
        if self._busy:
            return None
        low = fullname.lower()
        if "wanvideo" not in low or not low.endswith(self._suffix):
            return None
        # Resolve the real spec via the other finders, then wrap its loader.
        self._busy = True
        try:
            for finder in sys.meta_path:
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                try:
                    spec = find(fullname, path, target)
                except Exception:
                    spec = None
                if spec is not None and spec.loader is not None:
                    self._wrap(spec)
                    return spec
        finally:
            self._busy = False
        return None

    def _wrap(self, spec):
        loader = spec.loader
        orig_exec = loader.exec_module
        callback = self._callback

        def exec_module(module):
            orig_exec(module)
            try:
                callback(module)
            except Exception:
                import traceback
                traceback.print_exc()
            # One-shot: remove ourselves once the target module is handled.
            try:
                sys.meta_path.remove(_finder)
            except ValueError:
                pass

        loader.exec_module = exec_module


_finder = None


def install():
    global _installed, _finder
    if _installed:
        return

    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    # If WanVideoWrapper is already imported (re-init / load order), patch now.
    for name, mod in list(sys.modules.items()):
        if "wanvideo" in name.lower() and name.lower().endswith(_TARGET_SUFFIX):
            _patch_wanmodel(mod)
            _installed = True
            return

    _finder = _PostImportHook(_TARGET_SUFFIX, _patch_wanmodel)
    sys.meta_path.insert(0, _finder)
    _installed = True
    print(f"{TAG} armed; will neutralize WanVideo block swap on MPS when it loads.")
