"""Fix: psutil.virtual_memory() crashes on recent/beta macOS.

On macOS releases newer than the installed psutil build (notably macOS 26/27
developer betas), psutil's precompiled C extension fails almost every call:

    RuntimeError: host_statistics64(HOST_VM_INFO64) syscall failed:
                  (ipc/mig) array not large enough

Its `vm_statistics64` struct no longer matches the kernel's. ComfyUI calls
psutil.virtual_memory() on every node during a render (RAM-pressure cache +
model_management), so renders crash mid-way.

We probe psutil at startup; if it's unreliable, we replace psutil.virtual_memory
with a drop-in backed by the OS's own `vm_stat` + `sysctl hw.memsize`, which don't
use the broken syscall. On healthy machines we detect nothing wrong and do nothing.
"""

import collections
import re
import subprocess
import sys

TAG = "[AppleSilicon-FP8/psutil]"

_svmem = collections.namedtuple(
    "svmem",
    ["total", "available", "percent", "used", "free", "active", "inactive", "wired"],
)

_TOTAL = None  # hw.memsize is constant; cache it.


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
    fails = 0
    for _ in range(attempts):
        try:
            psutil.virtual_memory()
        except Exception:
            fails += 1
    return fails > 0


def install():
    if sys.platform != "darwin":
        return
    try:
        import psutil
    except Exception as e:
        print(f"{TAG} psutil not importable, skipping: {e}")
        return
    try:
        if not _is_broken(psutil):
            return  # healthy; leave psutil alone
        sample = _virtual_memory_vmstat()  # sanity-check before swapping
        psutil.virtual_memory = _virtual_memory_vmstat
        print(
            f"{TAG} psutil.virtual_memory() is broken on this OS — installed vm_stat "
            f"fallback (total={sample.total // (1024 ** 3)} GiB, "
            f"available={sample.available // (1024 ** 3)} GiB)."
        )
    except Exception as e:
        import traceback
        print(f"{TAG} failed to install fallback: {e}")
        traceback.print_exc()
