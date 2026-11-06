"""The Metal-extension builds must leave the C++ standard to torch (#34)."""
import importlib

import pytest

LOADERS = [
    "_patches.int8_ext.loader",
    "_patches.fp8_ext.loader",
    "_patches.int4_ext.loader",
]


@pytest.fixture(params=LOADERS, ids=lambda p: p.split(".")[1])
def loader(request):
    return importlib.import_module(request.param)


def test_missing_headers_stop_the_build_even_with_ninja(loader, tmp_path, monkeypatch, capsys):
    import os
    from _patches import _extbuild
    for flag in ("ASFP8_INT8_EXT", "ASFP8_FP8_EXT", "ASFP8_FP8_NATIVE", "ASFP8_INT4_EXT"):
        monkeypatch.setenv(flag, "1")
    monkeypatch.setattr(loader.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("ASFP8_BUILD_ROOT", str(tmp_path))
    monkeypatch.setattr(loader, "_tried", False)
    monkeypatch.setattr(loader, "_mod", None)
    monkeypatch.setattr(_extbuild, "_prereq_warned", False)
    real = os.path.exists
    monkeypatch.setattr(
        _extbuild.os.path, "exists",
        lambda p: False if str(p).endswith(os.path.join("torch", "extension.h")) else real(p))

    def must_not_build(*a, **k):
        raise AssertionError("a build was attempted without the torch headers")

    monkeypatch.setattr(loader, "_cpp_load_guarded", must_not_build)

    assert loader.module() is None
    out = capsys.readouterr().out
    assert "torch/include headers are missing" in out and "disabled" in out


def test_build_does_not_pin_the_cxx_standard(loader, tmp_path, monkeypatch):
    for flag in ("ASFP8_INT8_EXT", "ASFP8_FP8_EXT", "ASFP8_FP8_NATIVE", "ASFP8_INT4_EXT"):
        monkeypatch.setenv(flag, "1")
    monkeypatch.setattr(loader.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("ASFP8_BUILD_ROOT", str(tmp_path))
    monkeypatch.setattr(loader, "_tried", False)
    monkeypatch.setattr(loader, "_mod", None)
    seen = {}
    built = object()

    def fake_build(cpp_load, **kwargs):
        seen.update(kwargs)
        return built

    monkeypatch.setattr(loader, "_cpp_load_guarded", fake_build)

    assert loader.module() is built
    assert seen["build_directory"] == loader.build_dir_for(loader.__name__.split(".")[1])
    cflags = seen["extra_cflags"]
    assert "-ObjC++" in cflags
    assert not [f for f in cflags if f.startswith("-std=")], (
        f"{cflags} overrides the -std torch passes for its own headers"
    )
