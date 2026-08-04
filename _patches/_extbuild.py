"""Shared guards for the JIT Metal-extension builds (int8_ext / fp8_ext / int4_ext).

torch's cpp_extension.load takes no timeout and guards each build directory with a
FileBaton. Both facts have bitten us: building inside install() froze ComfyUI's
startup outright, and a lock left behind by a killed process stalls every later
build, because FileBaton.wait() spins on os.path.exists with no ceiling.
"""

import atexit
import os
import threading
import time

# Ceiling on one cold Metal-extension build, in seconds (ASFP8_EXT_BUILD_TIMEOUT;
# <=0 disables the watchdog). A cold build measures ~5 s on an M5 Max / macOS 27, so
# 600 s only fires when something is genuinely wedged.
BUILD_TIMEOUT_DEFAULT = 600.0


def _build_timeout():
    try:
        return float(os.environ.get("ASFP8_EXT_BUILD_TIMEOUT", BUILD_TIMEOUT_DEFAULT))
    except ValueError:
        return BUILD_TIMEOUT_DEFAULT


def _lock_path(build_dir):
    return os.path.join(build_dir or "", "lock")


def _clear_stale_lock(build_dir, tag=""):
    """Drop a build lock that no live build could still own.

    Age-gated deliberately: the build root is shared, so a young lock may belong to
    a concurrent build in another ComfyUI process. Removing that one would let two
    ninja runs write the same object files and produce a truncated extension.
    """
    lock = _lock_path(build_dir)
    try:
        age = time.time() - os.path.getmtime(lock)
    except OSError:
        return
    timeout = _build_timeout()
    threshold = timeout if timeout > 0 else BUILD_TIMEOUT_DEFAULT
    if age <= threshold:
        return
    try:
        os.unlink(lock)
        print(f"{tag} cleared a stale build lock ({age:.0f}s old); a previous build "
              f"was killed before it could release it.", flush=True)
    except OSError:
        pass


def _abandoned_lock_cleanup(build_dir, thread):
    """Cleanup for a build we stopped waiting on.

    torch releases the baton in a `finally` that an abandoned daemon thread never
    reaches, so without this every timeout strands a lock for the next run.
    """
    lock = _lock_path(build_dir)

    def cleanup():
        if thread.is_alive() and os.path.exists(lock):
            try:
                os.unlink(lock)
            except OSError:
                pass

    return cleanup


def _cpp_load_guarded(cpp_load, **kwargs):
    """`torch.utils.cpp_extension.load` with a wall-clock ceiling.

    There is no safe way to cancel torch's loader, so on expiry we raise and leave
    the build on a daemon thread; the caller degrades to "kernel unavailable" for
    this session.
    """
    timeout = _build_timeout()
    if timeout <= 0:
        return cpp_load(**kwargs)

    build_dir = kwargs.get("build_directory")
    result = {}

    def run():
        try:
            result["mod"] = cpp_load(**kwargs)
        except BaseException as e:  # re-raised on the calling thread
            result["err"] = e

    label = os.path.basename(build_dir or "") or "ext"
    t = threading.Thread(target=run, name=f"asfp8-{label}-build", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        atexit.register(_abandoned_lock_cleanup(build_dir, t))
        raise TimeoutError(
            f"Metal extension build still running after {timeout:g}s; giving up. "
            f"If this repeats, delete {build_dir} and restart. Raise "
            f"ASFP8_EXT_BUILD_TIMEOUT to allow a slower toolchain."
        )
    if "err" in result:
        raise result["err"]
    return result.get("mod")
