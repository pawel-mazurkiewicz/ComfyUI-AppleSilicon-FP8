"""JIT extension build dirs are keyed by the torch build; prerequisite failures are loud.

No MPS, no toolchain and no extension is built here: only the path helpers and the
one-shot warning run. `_nospace_torch_lib()` creates directories, so it is not called.
"""
import importlib
import os
import re
import sys

import pytest
import torch

from _patches import _extbuild

LOADERS = [
    "_patches.int8_ext.loader",
    "_patches.fp8_ext.loader",
    "_patches.int4_ext.loader",
]


def _patch_header_check(monkeypatch, exists):
    """Answer only the torch-header probe; every other os.path.exists stays real."""
    real = os.path.exists

    def fake(path):
        p = str(path)
        return exists if p.endswith(os.path.join("torch", "extension.h")) else real(path)

    monkeypatch.setattr(_extbuild.os.path, "exists", fake)


def test_build_dir_changes_with_torch_version(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.11.0")
    a = _extbuild.build_dir_for("int8_ext")
    monkeypatch.setattr(torch, "__version__", "2.12.0")
    b = _extbuild.build_dir_for("int8_ext")

    assert a != b, "a torch upgrade must not reuse the .so built against the old torch"
    prefix = _extbuild.build_root() + "/int8_ext-"
    assert a.startswith(prefix) and b.startswith(prefix)
    assert re.fullmatch(r"[0-9a-f]{10}", os.path.basename(b)[len("int8_ext-"):])


def test_build_root_env_override(monkeypatch):
    monkeypatch.setenv("ASFP8_BUILD_ROOT", "/tmp/x")
    assert _extbuild.build_dir_for("int8_ext").startswith("/tmp/x/int8_ext-")


def test_build_root_rejects_spaces(monkeypatch, capsys):
    monkeypatch.setenv("ASFP8_BUILD_ROOT", "/tmp/a b")
    assert _extbuild.build_root() == "/tmp/asfp8_build"
    assert "space" in capsys.readouterr().out


def test_warn_missing_prereqs_once(monkeypatch, capsys):
    monkeypatch.setattr(_extbuild.shutil, "which", lambda name: None)
    monkeypatch.setattr(_extbuild, "_prereq_warned", False)

    assert _extbuild.warn_missing_prereqs("[int8_ext]") is False
    assert _extbuild.warn_missing_prereqs("[int8_ext]") is False

    out = capsys.readouterr().out
    assert out.count("pip install ninja") == 1, out
    assert f'"{sys.executable}" -m pip install ninja' in out
    assert "\n" in out


def test_warn_missing_prereqs_names_the_header_fix(monkeypatch, capsys):
    monkeypatch.setattr(_extbuild.shutil, "which", lambda name: f"/usr/bin/{name}")
    _patch_header_check(monkeypatch, False)
    monkeypatch.setattr(_extbuild, "_prereq_warned", False)

    assert _extbuild.warn_missing_prereqs("[fp8_ext]") is False
    assert _extbuild.warn_missing_prereqs("[fp8_ext]") is False

    out = capsys.readouterr().out
    assert out.count("reinstall torch: the torch/include headers are missing") == 1, out
    assert "pip install ninja" not in out


def test_warn_missing_prereqs_ok(monkeypatch, capsys):
    monkeypatch.setattr(_extbuild.shutil, "which", lambda name: f"/usr/bin/{name}")
    _patch_header_check(monkeypatch, True)
    monkeypatch.setattr(_extbuild, "_prereq_warned", False)

    assert _extbuild.warn_missing_prereqs("[int4_ext]") is True
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("name", LOADERS)
def test_loaders_route_build_paths_through_extbuild(monkeypatch, name):
    monkeypatch.setenv("ASFP8_BUILD_ROOT", "/tmp/asfp8-test")
    loader = importlib.import_module(name)
    ext = name.split(".")[1]

    assert not hasattr(loader, "_NOSPACE_ROOT"), "one build root in _extbuild, not three"
    assert loader.build_root is _extbuild.build_root, "the torchlib symlink needs it too"
    assert loader.build_dir_for(ext).startswith(f"/tmp/asfp8-test/{ext}-")
