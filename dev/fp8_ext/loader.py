"""JIT loader for the experimental fp8-native matmul2d MPS extension.

Builds dev/fp8_ext/fp8_matmul2d.mm via torch.utils.cpp_extension.load (ObjC++,
linking Metal + Foundation). Opt-in (ASFP8_FP8_EXT=1) and fully guarded: returns
None if disabled, if the build toolchain is missing, or if compilation fails — the
caller falls back to the decode->MPS path. Cached after first successful load.

KNOWN SNAG (productionization TODO): cpp_extension does NOT quote the torch lib
`-L` path it auto-adds, so a SPACE in torch's install path (e.g. ".../IMPERIAL
SPACE/...") breaks the link step ("no such file or directory: 'SPACE/...'"). The
probe was validated by running through a no-space symlink to the venv
(`ln -s "<venv>" /tmp/asfp8_venv` + invoke `/tmp/asfp8_venv/bin/python`). A real
integration must build under a no-space path (auto-symlink the venv, or pre-build a
.metallib offline with `xcrun metal -std=metal4.1` and load via newLibraryWithURL).
"""

import os
import shutil

_mod = None
_tried = False


def available():
    return load() is not None


def load():
    global _mod, _tried
    if _tried:
        return _mod
    _tried = True

    if os.environ.get("ASFP8_FP8_EXT") != "1":
        print("[fp8_ext] disabled (set ASFP8_FP8_EXT=1 to enable the experimental build).")
        return None
    if shutil.which("xcrun") is None:
        print("[fp8_ext] no xcrun/toolchain; cannot build the extension.")
        return None

    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import load as cpp_load
    except Exception as e:
        print(f"[fp8_ext] torch cpp_extension unavailable: {e!r}")
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "fp8_matmul2d.mm")
    try:
        _mod = cpp_load(
            name="asfp8_fp8_matmul2d",
            sources=[src],
            extra_cflags=["-std=c++17", "-ObjC++"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            verbose=True,
        )
    except Exception as e:
        print(f"[fp8_ext] build failed: {e!r}")
        _mod = None
    return _mod
