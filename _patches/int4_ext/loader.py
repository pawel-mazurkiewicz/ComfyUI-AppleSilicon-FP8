"""JIT loader for the packed-int4 matmul2d MPS extension.

Opt-in via ASFP8_INT4_EXT=1; guarded and cached, and returns None on any failure. Same
space-in-path workaround as int8_ext/fp8_ext.
"""

import os
import shutil

from .._extbuild import (  # noqa: F401  (re-exported for the build-lock tests)
    _abandoned_lock_cleanup,
    _build_timeout,
    _clear_stale_lock,
    _cpp_load_guarded,
    build_dir_for,
    build_root,
    warn_missing_prereqs,
)

_mod = None
_tried = False


def _nospace_torch_lib():
    import torch.utils.cpp_extension as cpp
    real = cpp.TORCH_LIB_PATH
    if " " not in real:
        return real
    root = build_root()
    os.makedirs(root, exist_ok=True)
    link = os.path.join(root, "torchlib")
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

    if os.environ.get("ASFP8_INT4_EXT") != "1":
        return None
    if shutil.which("xcrun") is None:
        print("[int4_ext] no Metal toolchain (xcrun); int4-native disabled.")
        return None

    if shutil.which("ninja") is None:
        try:
            import ninja
            bin_dir = getattr(ninja, "BIN_DIR", None) or os.path.join(
                os.path.dirname(ninja.__file__), "data", "bin"
            )
            if os.path.isdir(bin_dir):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        except Exception as e:
            print(f"[int4_ext] ninja not importable ({e!r}); int4-native disabled.")
            return None
    if shutil.which("ninja") is None:
        warn_missing_prereqs("[int4_ext]")
        print("[int4_ext] int4-native disabled.", flush=True)
        return None

    try:
        import torch.utils.cpp_extension as cpp
        from torch.utils.cpp_extension import load as cpp_load
    except Exception as e:
        print(f"[int4_ext] cpp_extension unavailable: {e!r}")
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "int4_matmul2d.mm")
    build_dir = build_dir_for("int4_ext")
    os.makedirs(build_dir, exist_ok=True)
    _clear_stale_lock(build_dir, "[int4_ext]")

    print("[int4_ext] compiling the INT4 Metal kernel (first use; seconds to a few "
          "minutes depending on the toolchain) — not frozen, this resumes when the "
          "build ends.", flush=True)

    def _prepare():
        saved = cpp.TORCH_LIB_PATH
        cpp.TORCH_LIB_PATH = _nospace_torch_lib()
        return lambda: setattr(cpp, "TORCH_LIB_PATH", saved)

    try:
        _mod = _cpp_load_guarded(
            cpp_load,
            prepare=_prepare,
            name="asfp8_int4_matmul2d",
            sources=[src],
            extra_cflags=["-ObjC++"],  # no -std: torch sets its own
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            build_directory=build_dir,
            verbose=False,
        )
    except Exception as e:
        print(f"[int4_ext] build failed; int4-native disabled: {e!r}")
        _mod = None
    return _mod


def available():
    return module() is not None
