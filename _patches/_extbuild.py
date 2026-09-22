"""Shared guards for the JIT Metal-extension builds (int8_ext / fp8_ext / int4_ext).

torch's cpp_extension.load takes no timeout and guards each build directory with a
FileBaton whose wait() spins on os.path.exists forever, so a lock left by a killed
process stalls every later build.
"""

import atexit
import os
import threading
import time

# seconds for one cold build (ASFP8_EXT_BUILD_TIMEOUT; <=0 disables the watchdog)
BUILD_TIMEOUT_DEFAULT = 600.0


def _build_timeout():
    try:
        return float(os.environ.get("ASFP8_EXT_BUILD_TIMEOUT", BUILD_TIMEOUT_DEFAULT))
    except ValueError:
        return BUILD_TIMEOUT_DEFAULT


# serialises prepare/load/undo: prepare() mutates torch's module-level TORCH_LIB_PATH,
# so two overlapping workers would each restore the other's temporary value
_BUILD_LOCK = threading.Lock()


def _lock_path(build_dir):
    return os.path.join(build_dir or "", "lock")


def _clear_stale_lock(build_dir, tag=""):
    """Drop a build lock that no live build could still own.

    Age-gated: the build root is shared, so a young lock may belong to another ComfyUI
    process, and removing it would put two ninja runs in one directory.
    """
    lock = _lock_path(build_dir)
    try:
        age = time.time() - os.path.getmtime(lock)
    except OSError:
        return
    # twice the abandon timeout: we stop waiting at T but never kill the build, so a
    # lock just past T may still belong to a compile that is slow rather than wedged
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
    """Cleanup for a build we stopped waiting on, since an abandoned daemon thread
    never reaches torch's baton-releasing `finally`.

    `owned` is false when our thread is merely blocked in FileBaton.wait(): it is alive
    but holds nothing, and unlinking there would free another process's lock.
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

    torch's loader cannot be cancelled, so on expiry we raise and leave the build on a
    daemon thread. `prepare` runs on that thread and returns an undo callable: doing it
    around this call would restore the globals while an abandoned build still links.
    """
    timeout = _build_timeout()
    if timeout <= 0:
        with _BUILD_LOCK:
            undo = prepare() if prepare is not None else None
            try:
                return cpp_load(**kwargs)
            finally:
                if undo is not None:
                    undo()

    build_dir = kwargs.get("build_directory")
    # a lock already present isn't ours to clean up: we will be waiting on it
    pre_existing_lock = os.path.exists(_lock_path(build_dir))
    result = {}

    def run():
        undo = None
        with _BUILD_LOCK:
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
