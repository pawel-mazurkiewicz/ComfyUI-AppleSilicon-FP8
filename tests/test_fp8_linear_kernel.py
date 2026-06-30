import os

import pytest
import torch

from _patches.fp8_ext import loader


def _reset_loader(monkeypatch):
    monkeypatch.setattr(loader, "_tried", False, raising=False)
    monkeypatch.setattr(loader, "_mod", None, raising=False)


def test_loader_gated_off_without_any_flag(monkeypatch):
    monkeypatch.delenv("ASFP8_FP8_EXT", raising=False)
    monkeypatch.delenv("ASFP8_FP8_NATIVE", raising=False)
    _reset_loader(monkeypatch)
    assert loader.module() is None


def test_loader_memo_resets_between_flag_states(monkeypatch):
    # Off -> None, then flip NATIVE on with a fresh reset; the gate must re-evaluate.
    monkeypatch.delenv("ASFP8_FP8_EXT", raising=False)
    monkeypatch.delenv("ASFP8_FP8_NATIVE", raising=False)
    _reset_loader(monkeypatch)
    assert loader.module() is None
    monkeypatch.setenv("ASFP8_FP8_NATIVE", "1")
    _reset_loader(monkeypatch)
    # On non-MPS/no-toolchain CI this returns None via a *different* branch (xcrun/build),
    # NOT the env gate; that is still correct. On this M5 it builds and returns a module.
    _ = loader.module()  # must not raise; value depends on host


# --- Task 3: unit guards (no kernel needed; run everywhere) ---------------------
from _patches import fp8_linear_kernel_mps as patch


def test_install_noop_without_flag(monkeypatch):
    monkeypatch.delenv("ASFP8_FP8_NATIVE", raising=False)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


def test_eligibility_returns_none_no_kernel(monkeypatch):
    monkeypatch.setattr(patch, "_kernel", None, raising=False)
    class FakeLinear: weight = torch.zeros(8, 8, dtype=torch.float8_e4m3fn)
    assert patch._try_fp8_kernel_forward(FakeLinear(), torch.zeros(4, 8)) is None


def test_eligibility_rejects_non_fp8_weight(monkeypatch):
    # Kernel present but weight isn't a fp8 QuantizedTensor -> None (FAST handback to int8/orig).
    monkeypatch.setattr(patch, "_kernel", object(), raising=False)
    class FakeLinear: weight = torch.zeros(8, 8)   # plain, not QuantizedTensor
    assert patch._try_fp8_kernel_forward(FakeLinear(), torch.zeros(4, 8, device="cpu")) is None


def test_self_check_not_run_on_ineligible(monkeypatch):
    # BLOCKER 2 regression guard: a non-MPS/non-fp8 layer must NOT trigger the self-check.
    monkeypatch.setattr(patch, "_kernel", object(), raising=False)
    monkeypatch.setattr(patch, "_self_checked", False, raising=False)
    calls = {"n": 0}
    def boom():
        calls["n"] += 1
        return True
    monkeypatch.setattr(patch, "_self_check", boom, raising=False)
    class FakeLinear: weight = torch.zeros(8, 8)   # ineligible
    assert patch._try_fp8_kernel_forward(FakeLinear(), torch.zeros(4, 8, device="cpu")) is None
    assert calls["n"] == 0, "self-check ran on an ineligible layer (BLOCKER 2 regression)"
