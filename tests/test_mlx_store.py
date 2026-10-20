import json
import os

import pytest

from _patches import _mlx_store

hub = pytest.importorskip("huggingface_hub")

REPO = "mlx-community/Qwen3.5-9B-4bit"


class _Hub:
    """snapshot_download stand-in: cached repos answer local_files_only, the rest 'download'."""
    def __init__(self, cached=None):
        self.cached = cached
        self.calls = []

    def __call__(self, repo, local_files_only=False, local_dir=None):
        self.calls.append((repo, local_files_only, local_dir))
        if local_files_only:
            if self.cached is None:
                raise FileNotFoundError(repo)
            return self.cached
        return local_dir or "/hf-cache/" + repo


def _model_dir(path, shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
               present=None, single=False):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    if single:
        (path / "model.safetensors").write_bytes(b"")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}))
    for s in shards if present is None else present:
        (path / s).write_bytes(b"")
    return path


@pytest.fixture
def comfy_dir(tmp_path, monkeypatch):
    base = tmp_path / "text_encoders"
    base.mkdir()
    monkeypatch.setattr(_mlx_store, "_comfy_dirs", lambda: [str(base), str(tmp_path / "clip")])
    return base


def test_local_path_used_as_is(tmp_path, monkeypatch):
    fake = _Hub()
    monkeypatch.setattr(hub, "snapshot_download", fake)
    assert _mlx_store.resolve(str(tmp_path), "T") == str(tmp_path)
    assert fake.calls == []


def test_existing_copy_in_text_encoders(comfy_dir, monkeypatch):
    path = _model_dir(comfy_dir / "mlx" / "mlx-community" / "Qwen3.5-9B-4bit")
    fake = _Hub()
    monkeypatch.setattr(hub, "snapshot_download", fake)
    assert _mlx_store.resolve(REPO, "T") == str(path)
    assert fake.calls == []


def test_hf_cache_copy_is_reused(comfy_dir, tmp_path, monkeypatch):
    cached = str(_model_dir(tmp_path / "snap"))
    fake = _Hub(cached=cached)
    monkeypatch.setattr(hub, "snapshot_download", fake)
    assert _mlx_store.resolve(REPO, "T") == cached
    assert fake.calls == [(REPO, True, None)]


def test_partial_copy_resumes_download(comfy_dir, tmp_path, monkeypatch):
    path = _model_dir(comfy_dir / "mlx" / "mlx-community" / "Qwen3.5-9B-4bit",
                      present=["model-00001-of-00002.safetensors"])
    fake = _Hub(cached=str(_model_dir(tmp_path / "snap", present=[])))
    monkeypatch.setattr(hub, "snapshot_download", fake)
    assert _mlx_store.resolve(REPO, "T") == str(path)
    assert fake.calls[-1] == (REPO, False, str(path))


def test_stale_index_with_single_file_is_complete(tmp_path):
    assert _mlx_store._complete(str(_model_dir(tmp_path / "m", present=[], single=True)))
    assert not _mlx_store._complete(str(_model_dir(tmp_path / "n", present=[])))


def test_downloads_into_first_text_encoders_dir(comfy_dir, monkeypatch):
    fake = _Hub()
    monkeypatch.setattr(hub, "snapshot_download", fake)
    want = os.path.join(str(comfy_dir), "mlx", "mlx-community", "Qwen3.5-9B-4bit")
    assert _mlx_store.resolve(REPO, "T") == want
    assert fake.calls[-1] == (REPO, False, want)


def test_hf_cache_download_outside_comfy(monkeypatch):
    monkeypatch.setattr(_mlx_store, "_comfy_dirs", lambda: [])
    fake = _Hub()
    monkeypatch.setattr(hub, "snapshot_download", fake)
    assert _mlx_store.resolve(REPO, "T") == "/hf-cache/" + REPO
    assert fake.calls[-1] == (REPO, False, None)
