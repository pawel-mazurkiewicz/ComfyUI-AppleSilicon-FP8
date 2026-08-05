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
    # Twice the abandon timeout, not once: we give up waiting at T but never kill
    # the build, so a lock just past T may still belong to a compile that is slow
    # rather than wedged. Being too patient only costs one more stall on a machine
    # that is already broken; being too eager puts two ninja runs in one directory.
    timeout = _build_timeout()
    threshold = 2 * (timeout if timeout > 0 else BUILD_TIMEOUT_DEFAULT)
    if age <= threshold:
        return
    try:
        os.unlink(lock)
        print(f"{tag} cleared a stale build lock ({age:.0f}s old); a previous build "
              f"was killed before it could release it.", flush=True)
    except OSError:
        pass


def _abandoned_lock_cleanup(build_dir, thread, owned=True):
    """Cleanup for a build we stopped waiting on.

    torch releases the baton in a `finally` that an abandoned daemon thread never
    reaches, so without this every timeout strands a lock for the next run.

    `owned` guards the case a live thread cannot distinguish on its own: a thread
    blocked in FileBaton.wait() is alive but holds nothing, and unlinking there
    would free another process's lock and fail its build on release.
    """
    lock = _lock_path(build_dir)

    def cleanup():
        if owned and thread.is_alive() and os.path.exists(lock):
            try:
                os.unlink(lock)
            except OSError:
                pass

    return cleanup


def _cpp_load_guarded(cpp_load, prepare=None, **kwargs):
    """`torch.utils.cpp_extension.load` with a wall-clock ceiling.

    There is no safe way to cancel torch's loader, so on expiry we raise and leave
    the build on a daemon thread; the caller degrades to "kernel unavailable" for
    this session.

    `prepare` runs on the build thread and returns an undo callable. Globals the
    build needs (torch's TORCH_LIB_PATH) must be set and restored there: doing it
    around this call would restore them while an abandoned build is still linking.
    """
    timeout = _build_timeout()
    if timeout <= 0:
        undo = prepare() if prepare is not None else None
        try:
            return cpp_load(**kwargs)
        finally:
            if undo is not None:
                undo()

    build_dir = kwargs.get("build_directory")
    # A lock already present isn't ours: our thread will be waiting on it, not
    # holding it, so we must not clean it up on the way out.
    pre_existing_lock = os.path.exists(_lock_path(build_dir))
    result = {}

    def run():
        undo = None
        try:
            if prepare is not None:
                undo = prepare()
            result["mod"] = cpp_load(**kwargs)
        except BaseException as e:  # re-raised on the calling thread
            result["err"] = e
        finally:
            if undo is not None:
                try:
                    undo()
                except Exception:
                    pass

    label = os.path.basename(build_dir or "") or "ext"
    t = threading.Thread(target=run, name=f"asfp8-{label}-build", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        atexit.register(
            _abandoned_lock_cleanup(build_dir, t, owned=not pre_existing_lock)
        )
        raise TimeoutError(
            f"Metal extension build still running after {timeout:g}s; giving up. "
            f"If this repeats, delete {build_dir} and restart. Raise "
            f"ASFP8_EXT_BUILD_TIMEOUT to allow a slower toolchain."
        )
    if "err" in result:
        raise result["err"]
    return result.get("mod")
