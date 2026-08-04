"""JIT loader for the bit-exact int8 matmul2d MPS extension.

Builds _patches/int8_ext/int8_gemm.mm via torch.utils.cpp_extension.load (ObjC++,
Metal 4.1). DEFAULT ON where capable (Tier B: M5 / Metal 4.1 + ninja, via the shared
three-state gate); guarded, cached; returns None on any failure so callers fall back
to the fp32/bf16 _int_mm path.

Space-in-path workaround (same as fp8_ext): cpp_extension emits torch's lib `-L`
UNQUOTED, so a space in torch's install path (".../IMPERIAL SPACE/...") breaks the
link. We point the linker at a no-space symlink to torch/lib and build under a
no-space directory. Best-effort, reverted on failure.
"""

import os
import shutil

from .._extbuild import (  # noqa: F401  (re-exported for the build-lock tests)
    _abandoned_lock_cleanup,
    _build_timeout,
    _clear_stale_lock,
    _cpp_load_guarded,
)

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

    # DEFAULT ON where capable (Tier B: M5/Metal-4.1 + ninja). Three-state gate:
    # unset -> on iff tier_b_ready; ASFP8_INT8_EXT=off -> off; =1 -> force the build.
    from .. import _caps
    if not _caps.resolve("ASFP8_INT8_EXT", default_on=True, cap=_caps.tier_b_ready):
        return None
    if shutil.which("xcrun") is None:
        print("[int8_ext] no Metal toolchain (xcrun); int8-native disabled.")
        return None

    # torch's cpp_extension needs the `ninja` *binary* on PATH (not just the
    # python module). When ComfyUI is launched from the Desktop app, PATH often
    # lacks homebrew/venv bins, so add the ninja package's bundled binary dir.
    if shutil.which("ninja") is None:
        try:
            import ninja  # provides a bundled `ninja` executable
            bin_dir = getattr(ninja, "BIN_DIR", None) or os.path.join(
                os.path.dirname(ninja.__file__), "data", "bin"
            )
            if os.path.isdir(bin_dir):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        except Exception as e:
            print(f"[int8_ext] ninja not importable ({e!r}); int8-native disabled.")
            return None
    if shutil.which("ninja") is None:
        print("[int8_ext] ninja binary not found on PATH; int8-native disabled "
              "(pip install ninja into the ComfyUI venv).")
        return None

    try:
        import torch.utils.cpp_extension as cpp
        from torch.utils.cpp_extension import load as cpp_load
    except Exception as e:
        print(f"[int8_ext] cpp_extension unavailable: {e!r}")
        return None

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "int8_gemm.mm")
    build_dir = os.path.join(_NOSPACE_ROOT, "int8_ext")
    os.makedirs(build_dir, exist_ok=True)
    _clear_stale_lock(build_dir, "[int8_ext]")

    print("[int8_ext] compiling the INT8 Metal kernel (first use; seconds to a few "
          "minutes depending on the toolchain) — not frozen, this resumes when the "
          "build ends.", flush=True)

    saved = cpp.TORCH_LIB_PATH
    try:
        cpp.TORCH_LIB_PATH = _nospace_torch_lib()
        _mod = _cpp_load_guarded(
            cpp_load,
            name="asfp8_int8_gemm",
            sources=[src],
            extra_cflags=["-std=c++17", "-ObjC++"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            build_directory=build_dir,
            verbose=False,
        )
    except Exception as e:
        print(f"[int8_ext] build failed; int8-native disabled: {e!r}")
        _mod = None
    finally:
        cpp.TORCH_LIB_PATH = saved
    return _mod


def available():
    return module() is not None
