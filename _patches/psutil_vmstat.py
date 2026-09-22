"""Fix: psutil.virtual_memory() crashes on recent/beta macOS.

psutil's `vm_statistics64` struct stops matching the kernel's, so nearly every call
raises. Probe at startup and, if it is unreliable, swap in a drop-in backed by `vm_stat`
and `sysctl hw.memsize`, which don't use the broken syscall.
"""

import collections
import os
import re
import subprocess
import sys

TAG = "[AppleSilicon-FP8/psutil]"

_svmem = collections.namedtuple(
    "svmem",
    ["total", "available", "percent", "used", "free", "active", "inactive", "wired"],
)

_TOTAL = None  # hw.memsize is constant, so cache it


def _sysctl_int(name):
    return int(subprocess.check_output(["sysctl", "-n", name]).strip())


def _vm_stat():
    out = subprocess.check_output(["vm_stat"]).decode()
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    d = {}
    for line in out.splitlines():
        m = re.match(r'"?([^":]+)"?:\s+(\d+)\.', line)
        if m:
            d[m.group(1).strip()] = int(m.group(2))
    return page, d


def _virtual_memory_vmstat():
    """Drop-in for psutil.virtual_memory() mirroring psutil's macOS math."""
    global _TOTAL
    if _TOTAL is None:
        _TOTAL = _sysctl_int("hw.memsize")
    total = _TOTAL
    page, d = _vm_stat()
    free = d.get("Pages free", 0) * page
    active = d.get("Pages active", 0) * page
    inactive = d.get("Pages inactive", 0) * page
    wired = d.get("Pages wired down", 0) * page
    speculative = d.get("Pages speculative", 0) * page
    avail = inactive + free
    used = active + wired
    free -= speculative
    percent = round((total - avail) / total * 100, 1) if total else 0.0
    return _svmem(total, avail, percent, used, free, active, inactive, wired)


def _is_broken(psutil, attempts=24):
    """True only if psutil.virtual_memory() fails for a clear majority of calls.

    The bug fails nearly every call, so a majority threshold can't mistake one hiccup
    on a healthy OS for it.
    """
    fails = 0
    for _ in range(attempts):
        try:
            psutil.virtual_memory()
        except Exception:
            fails += 1
    return fails > attempts // 2


def _mode():
    """Read APPLESILICON_FP8_PSUTIL: 'auto' (default), 'on'/'force', or 'off'."""
    v = os.environ.get("APPLESILICON_FP8_PSUTIL", "auto").strip().lower()
    if v in ("0", "off", "false", "no", "disable", "disabled"):
        return "off"
    if v in ("1", "on", "true", "yes", "force", "forced"):
        return "on"
    return "auto"


def install():
    if sys.platform != "darwin":
        return
    mode = _mode()
    if mode == "off":
        return
    try:
        import psutil
    except Exception as e:
        print(f"{TAG} psutil not importable, skipping: {e}")
        return
    try:
        if mode == "auto" and not _is_broken(psutil):
            return  # healthy: leave psutil completely untouched
        sample = _virtual_memory_vmstat()  # sanity-check before swapping
        psutil.virtual_memory = _virtual_memory_vmstat
        why = (
            "forced via APPLESILICON_FP8_PSUTIL"
            if mode == "on"
            else "psutil.virtual_memory() is broken on this OS"
        )
        print(
            f"{TAG} {why} — installed vm_stat fallback "
            f"(total={sample.total // (1024 ** 3)} GiB, "
            f"available={sample.available // (1024 ** 3)} GiB)."
        )
    except Exception as e:
        import traceback
        print(f"{TAG} failed to install fallback: {e}")
        traceback.print_exc()
