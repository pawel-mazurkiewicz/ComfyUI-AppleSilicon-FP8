"""Metal-extension build locks must not wedge a later run.

torch releases each FileBaton in a `finally` that a killed ComfyUI never reaches, and
FileBaton.wait() spins unbounded. Run against all three shipped loaders.
"""
import importlib
import os
import threading
import time

import pytest

LOADERS = [
    "_patches.int8_ext.loader",
    "_patches.fp8_ext.loader",
    "_patches.int4_ext.loader",
]


@pytest.fixture(params=LOADERS, ids=lambda p: p.split(".")[1])
def loader(request):
    return importlib.import_module(request.param)


def _plant_lock(d, age_seconds=0.0):
    lock = os.path.join(str(d), "lock")
    with open(lock, "w"):
        pass
    if age_seconds:
        t = time.time() - age_seconds
        os.utime(lock, (t, t))
    return lock


def test_build_is_watchdogged(loader):
    """A bare cpp_load can block forever; every loader must bound it."""
    assert hasattr(loader, "_cpp_load_guarded")


def test_stale_lock_is_cleared(loader, tmp_path, monkeypatch):
    monkeypatch.setenv("ASFP8_EXT_BUILD_TIMEOUT", "60")
    lock = _plant_lock(tmp_path, age_seconds=600)

    loader._clear_stale_lock(str(tmp_path))

    assert not os.path.exists(lock), "an abandoned lock must not wedge the next build"


def test_live_build_lock_is_preserved(loader, tmp_path, monkeypatch):
    """A young lock is preserved: it may belong to a live build in another process."""
    monkeypatch.setenv("ASFP8_EXT_BUILD_TIMEOUT", "600")
    lock = _plant_lock(tmp_path)

    loader._clear_stale_lock(str(tmp_path))

    assert os.path.exists(lock)


def test_timeout_message_points_at_the_build_dir(loader, tmp_path, monkeypatch):
    """The timeout message points at the build dir, not at disabling the watchdog."""
    monkeypatch.setenv("ASFP8_EXT_BUILD_TIMEOUT", "0.3")
    release = threading.Event()

    def stalls(**kwargs):
        release.wait(30)

    try:
        with pytest.raises(TimeoutError) as excinfo:
            loader._cpp_load_guarded(stalls, build_directory=str(tmp_path))
    finally:
        release.set()

    message = str(excinfo.value)
    assert str(tmp_path) in message, "must name the directory the user should delete"
    assert "set it to 0" not in message


def test_abandoned_build_cleanup_drops_the_lock(loader, tmp_path):
    """The orphaned build thread can never release its own lock, so we do it."""
    lock = _plant_lock(tmp_path)
    release = threading.Event()
    thread = threading.Thread(target=release.wait, args=(30,), daemon=True)
    thread.start()
    try:
        loader._abandoned_lock_cleanup(str(tmp_path), thread)()
        assert not os.path.exists(lock)
    finally:
        release.set()


def test_lock_from_a_slow_but_live_build_is_preserved(loader, tmp_path, monkeypatch):
    """A lock just past the timeout is preserved: we abandon a build but never kill it."""
    monkeypatch.setenv("ASFP8_EXT_BUILD_TIMEOUT", "60")
    lock = _plant_lock(tmp_path, age_seconds=90)

    loader._clear_stale_lock(str(tmp_path))

    assert os.path.exists(lock)


def test_cleanup_leaves_a_lock_we_never_owned(loader, tmp_path):
    """Cleanup leaves a lock we never owned: a thread in FileBaton.wait() holds nothing."""
    lock = _plant_lock(tmp_path)
    release = threading.Event()
    thread = threading.Thread(target=release.wait, args=(30,), daemon=True)
    thread.start()
    try:
        loader._abandoned_lock_cleanup(str(tmp_path), thread, owned=False)()
        assert os.path.exists(lock)
    finally:
        release.set()


def test_concurrent_builds_do_not_corrupt_the_prepared_global(loader, monkeypatch):
    """Concurrent builds don't corrupt TORCH_LIB_PATH, which prepare() mutates globally."""
    monkeypatch.setenv("ASFP8_EXT_BUILD_TIMEOUT", "30")
    state = {"path": "ORIGINAL"}
    # released only once both threads are in place, so the overlap is real rather than
    # two runs that happened to serialise
    both_ready = threading.Barrier(2, timeout=30)
    tally = threading.Lock()
    active = {"now": 0, "max": 0}

    def prepare():
        saved = state["path"]
        state["path"] = "TEMP"
        with tally:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)          # widen the window both workers race through

        def undo():
            with tally:
                active["now"] -= 1
            state["path"] = saved

        return undo

    def slow_load(**kwargs):
        time.sleep(0.05)

    def worker():
        both_ready.wait()         # must be outside the build lock, or we deadlock
        loader._cpp_load_guarded(slow_load, prepare=prepare)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)

    assert not any(t.is_alive() for t in threads), "a build worker never finished"
    assert active["max"] == 1, (
        f"{active['max']} workers held the prepared global at once; prepare, "
        "load and undo must be mutually exclusive"
    )
    assert state["path"] == "ORIGINAL", (
        f"TORCH_LIB_PATH left as {state['path']!r}; a worker restored another "
        "worker's temporary value"
    )
