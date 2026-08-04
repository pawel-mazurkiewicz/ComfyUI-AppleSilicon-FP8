"""Metal-extension build locks must not wedge a later run.

torch guards each build directory with a FileBaton whose lock is released in a
`finally`. A killed ComfyUI -- or our own watchdog abandoning a build on a daemon
thread -- never runs it, and FileBaton.wait() is an unbounded spin, so the lock
stalls every subsequent build until someone deletes it by hand.

Every shipped loader that calls cpp_extension.load needs the same treatment, so
these run against all three.
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
    """A young lock may belong to a live build in another ComfyUI process.

    /tmp/asfp8_build is a shared path; clearing it blindly would let two ninja
    runs write the same .so concurrently.
    """
    monkeypatch.setenv("ASFP8_EXT_BUILD_TIMEOUT", "600")
    lock = _plant_lock(tmp_path)

    loader._clear_stale_lock(str(tmp_path))

    assert os.path.exists(lock)


def test_timeout_message_points_at_the_build_dir(loader, tmp_path, monkeypatch):
    """The old text suggested ASFP8_EXT_BUILD_TIMEOUT=0, which disables the
    watchdog entirely and turns a bounded stall into an unbounded hang."""
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
