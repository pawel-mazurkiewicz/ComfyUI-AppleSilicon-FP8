"""Switch: neutralize ComfyUI-WanVideoWrapper block swap on MPS.

Block swap streams transformer blocks back from the CPU using CUDA events to sync the
async copy, which doesn't hold on MPS, so a block computes with parameters still on the
CPU. Unified memory makes the swap pure downside anyway, so wrap WanModel.forward and
make every block resident with the offload flags cleared. ASFP8_NEUTRALIZE_BLOCKSWAP=off
disables. WanVideoWrapper imports after us, hence the post-import hook.
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
            if main_dev is None:
                is_mps = torch.backends.mps.is_available()
            if is_mps:
                # clear every offload knob so the real forward never streams
                self.blocks_to_swap = 0
                if hasattr(self, "vace_blocks_to_swap"):
                    self.vace_blocks_to_swap = 0
                if hasattr(self, "prefetch_blocks"):
                    self.prefetch_blocks = 0
                self.offload_txt_emb = False
                self.offload_img_emb = False
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
        # resolve the real spec via the other finders, then wrap its loader
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
            # one-shot: remove ourselves once the target module is handled
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

    # already imported (re-init, or a different load order): patch it now
    for name, mod in list(sys.modules.items()):
        if "wanvideo" in name.lower() and name.lower().endswith(_TARGET_SUFFIX):
            _patch_wanmodel(mod)
            _installed = True
            return

    _finder = _PostImportHook(_TARGET_SUFFIX, _patch_wanmodel)
    sys.meta_path.insert(0, _finder)
    _installed = True
    print(f"{TAG} armed; will neutralize WanVideo block swap on MPS when it loads.")
