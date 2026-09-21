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


def test_build_does_not_pin_the_cxx_standard(loader, tmp_path, monkeypatch):
    for flag in ("ASFP8_INT8_EXT", "ASFP8_FP8_EXT", "ASFP8_FP8_NATIVE", "ASFP8_INT4_EXT"):
        monkeypatch.setenv(flag, "1")
    monkeypatch.setattr(loader.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(loader, "_NOSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(loader, "_tried", False)
    monkeypatch.setattr(loader, "_mod", None)
    seen = {}
    built = object()

    def fake_build(cpp_load, **kwargs):
        seen.update(kwargs)
        return built

    monkeypatch.setattr(loader, "_cpp_load_guarded", fake_build)

    assert loader.module() is built
    cflags = seen["extra_cflags"]
    assert "-ObjC++" in cflags
    assert not [f for f in cflags if f.startswith("-std=")], (
        f"{cflags} overrides the -std torch passes for its own headers"
    )
