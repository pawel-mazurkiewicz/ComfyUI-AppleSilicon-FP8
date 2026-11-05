import sys
import types

import pytest
import torch

from _patches import free_mem_mps

GiB = 1 << 30


@pytest.fixture
def env(monkeypatch):
    """Fake comfy modules plus controllable torch.mps counters."""
    calls = []
    state = {"avail": 10 * GiB, "driver": 30 * GiB, "current": 22 * GiB}

    def original(dev=None, torch_free_too=False):
        calls.append((dev, torch_free_too))
        return ("orig", "orig") if torch_free_too else "orig"

    mm = types.ModuleType("comfy.model_management")
    mm.get_free_memory = original
    mm.get_torch_device = lambda: torch.device("mps")
    sm = types.ModuleType("comfy.system_memory")
    sm.virtual_memory_available = lambda: state["avail"]
    pkg = types.ModuleType("comfy")
    pkg.model_management, pkg.system_memory = mm, sm
    for name, mod in (("comfy", pkg), ("comfy.model_management", mm), ("comfy.system_memory", sm)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: state["driver"])
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: state["current"])
    monkeypatch.setattr(free_mem_mps, "_supported", lambda: True)
    monkeypatch.setattr(free_mem_mps, "_failed", False, raising=False)
    monkeypatch.delenv("ASFP8_FREE_MEM", raising=False)
    monkeypatch.delitem(sys.modules, "mlx.core", raising=False)
    mm.calls, mm.state = calls, state
    return mm


def test_mps_adds_reusable_pool(env):
    assert free_mem_mps.install()
    mps = torch.device("mps")
    assert env.get_free_memory(mps, True) == (18 * GiB, 8 * GiB)
    assert env.get_free_memory(mps) == 18 * GiB
    assert env.get_free_memory(None) == 18 * GiB
    assert env.calls == []


def test_pool_never_negative(env):
    env.state.update(driver=5 * GiB, current=9 * GiB)
    free_mem_mps.install()
    assert env.get_free_memory(torch.device("mps"), True) == (10 * GiB, 0)


def test_cpu_device_untouched(env):
    free_mem_mps.install()
    assert env.get_free_memory(torch.device("cpu"), True) == ("orig", "orig")
    assert env.calls == [(torch.device("cpu"), True)]


def test_off_switch(env, monkeypatch):
    free_mem_mps.install()
    monkeypatch.setenv("ASFP8_FREE_MEM", "off")
    assert env.get_free_memory(torch.device("mps")) == "orig"


def test_off_switch_at_install_leaves_original(env, monkeypatch):
    original = env.get_free_memory
    monkeypatch.setenv("ASFP8_FREE_MEM", "0")
    assert not free_mem_mps.install()
    assert env.get_free_memory is original


def test_install_idempotent(env):
    free_mem_mps.install()
    wrapped = env.get_free_memory
    free_mem_mps.install()
    assert env.get_free_memory is wrapped
    assert wrapped._asfp8_wrapped
    env.get_free_memory(torch.device("cpu"))
    assert len(env.calls) == 1


def test_unsupported_platform_does_not_wrap(env, monkeypatch):
    original = env.get_free_memory
    monkeypatch.setattr(free_mem_mps, "_supported", lambda: False)
    assert not free_mem_mps.install()
    assert env.get_free_memory is original


def test_mlx_memory_is_not_counted_as_pool(env, monkeypatch):
    # MLX buffers raise the device-wide driver total; only torch's own pool may count.
    mlx = types.ModuleType("mlx.core")
    mlx.get_active_memory = lambda: 6 * GiB
    mlx.get_cache_memory = lambda: 1 * GiB
    monkeypatch.setitem(sys.modules, "mlx.core", mlx)
    env.state.update(driver=40 * GiB, current=22 * GiB)
    free_mem_mps.install()
    assert env.get_free_memory(torch.device("mps"), True) == (21 * GiB, 11 * GiB)


def test_mlx_larger_than_driver_gap_clamps_to_zero(env, monkeypatch):
    mlx = types.ModuleType("mlx.core")
    mlx.get_active_memory = lambda: 20 * GiB
    mlx.get_cache_memory = lambda: 0
    monkeypatch.setitem(sys.modules, "mlx.core", mlx)
    free_mem_mps.install()
    assert env.get_free_memory(torch.device("mps"), True) == (10 * GiB, 0)


def test_counter_failure_falls_back_to_original(env, monkeypatch):
    def boom():
        raise RuntimeError("no mps")

    free_mem_mps.install()
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", boom)
    assert env.get_free_memory(torch.device("mps")) == "orig"


def test_counter_failure_latches_once(env, monkeypatch, capsys):
    seen = []

    def boom():
        seen.append(1)
        raise RuntimeError("no mps")

    free_mem_mps.install()
    capsys.readouterr()
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", boom)
    mps = torch.device("mps")
    assert env.get_free_memory(mps) == "orig"
    assert env.get_free_memory(mps) == "orig"
    assert len(seen) == 1
    assert capsys.readouterr().out.count(free_mem_mps.TAG) == 1


def test_mlx_metal_namespace_fallback(env, monkeypatch):
    metal = types.SimpleNamespace(get_active_memory=lambda: 6 * GiB, get_cache_memory=lambda: 1 * GiB)
    mlx = types.ModuleType("mlx.core")
    mlx.metal = metal
    monkeypatch.setitem(sys.modules, "mlx.core", mlx)
    env.state.update(driver=40 * GiB, current=22 * GiB)
    free_mem_mps.install()
    assert env.get_free_memory(torch.device("mps"), True) == (21 * GiB, 11 * GiB)
