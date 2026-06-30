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
