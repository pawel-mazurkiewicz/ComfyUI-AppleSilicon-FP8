"""JIT loader for the fp8-native matmul2d MPS extension (patch #15 backend).

Builds _patches/fp8_ext/fp8_matmul2d.mm via torch.utils.cpp_extension.load (ObjC++,
Metal 4.1). Opt-in (ASFP8_FP8_EXT=1), guarded, cached; returns None on any failure so
the caller falls back to the decode path.

Space-in-path workaround: cpp_extension emits the torch lib `-L` UNQUOTED, so a space
in torch's install path (e.g. ".../IMPERIAL SPACE/...") breaks the link. We point the
linker at a no-space symlink to torch/lib by overriding cpp_extension.TORCH_LIB_PATH,
and build under a no-space directory. Both are best-effort and reverted on failure.
"""

import os
import shutil

_mod = None
_tried = False

_NOSPACE_ROOT = "/tmp/asfp8_build"


def _nospace_torch_lib():
    """Return a no-space path to torch/lib (via symlink if the real path has a space)."""
    import torch.utils.cpp_extension as cpp
    real = cpp.TORCH_LIB_PATH
    if " " not in real:
        return real
    os.makedirs(_NOSPACE_ROOT, exist_ok=True)
    link = os.path.join(_NOSPACE_ROOT, "torchlib")
    try:
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(real, link)
    except OSError:
        return real
    return link


def module():
    global _mod, _tried
    if _tried:
        return _mod
    _tried = True

    if os.environ.get("ASFP8_FP8_EXT") != "1":
        return None
    if shutil.which("xcrun") is None:
        print("[fp8_ext] no Metal toolchain (xcrun); fp8-native disabled.")
        return None
    try:
        import torch.utils.cpp_extension as cpp
        from torch.utils.cpp_extension import load as cpp_load
    except Exception as e:
        print(f"[fp8_ext] cpp_extension unavailable: {e!r}")
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "fp8_matmul2d.mm")
    build_dir = os.path.join(_NOSPACE_ROOT, "ext")
    os.makedirs(build_dir, exist_ok=True)

    saved = cpp.TORCH_LIB_PATH
    try:
        cpp.TORCH_LIB_PATH = _nospace_torch_lib()
        _mod = cpp_load(
            name="asfp8_fp8_matmul2d",
            sources=[src],
            extra_cflags=["-std=c++17", "-ObjC++"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            build_directory=build_dir,
            verbose=False,
        )
    except Exception as e:
        print(f"[fp8_ext] build failed; fp8-native disabled: {e!r}")
        _mod = None
    finally:
        cpp.TORCH_LIB_PATH = saved
    return _mod


def available():
    return module() is not None
