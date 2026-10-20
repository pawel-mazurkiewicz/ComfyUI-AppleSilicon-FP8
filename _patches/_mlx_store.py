"""Where the mlx_textgen backends keep MLX models (ComfyUI's text_encoders folder, under
mlx/), and the one MLX model kept resident across them."""

import gc
import json
import os
import struct

_TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json")

_LOADED = {}  # repo -> loaded model; at most one


def _comfy_dirs():
    try:
        import folder_paths
        return list(folder_paths.get_folder_paths("text_encoders"))
    except Exception:
        return []


def _local_dir(base, repo):
    return os.path.join(base, "mlx", *repo.split("/"))


def _safetensors_whole(path):
    """The file is at least as long as its own header says (catches truncated copies)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if 8 + n > size:
                return False
            header = json.loads(f.read(n))
        end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
        return 8 + n + end <= size
    except (OSError, ValueError, KeyError, TypeError, IndexError, struct.error):
        return False


def _weight_files(path):
    # some repos ship a stale index next to a single model.safetensors
    if os.path.isfile(os.path.join(path, "model.safetensors")):
        return ["model.safetensors"]
    try:
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return []


def _complete(path):
    """Config, tokenizer and every weight file whole: an interrupted download is resumed, not loaded."""
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    if not os.path.isfile(os.path.join(path, "tokenizer_config.json")):
        return False
    if not any(os.path.isfile(os.path.join(path, t)) for t in _TOKENIZER_FILES):
        return False
    weights = _weight_files(path)
    return bool(weights) and all(_safetensors_whole(os.path.join(path, w)) for w in weights)


def resolve(repo, tag):
    """Local folder for `repo`: as given, text_encoders/mlx/, HF cache, else download there."""
    if os.path.isdir(repo):
        if not _complete(repo):
            raise RuntimeError(f"{repo} is missing config, tokenizer or whole weight files")
        return repo
    dirs = _comfy_dirs()
    for base in dirs:
        path = _local_dir(base, repo)
        if _complete(path):
            return path

    from huggingface_hub import snapshot_download
    try:
        cached = snapshot_download(repo, local_files_only=True)
        if _complete(cached):
            return cached
    except Exception:
        pass
    if not dirs:
        return snapshot_download(repo)
    path = _local_dir(dirs[0], repo)
    print(f"{tag} downloading {repo} to {path}")
    return snapshot_download(repo, local_dir=path)


def _release():
    if not _LOADED:
        return
    _LOADED.clear()
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def load_cached(repo, tag, load):
    """`load(path)` for `repo`, keeping one MLX model resident: switching repos frees the last."""
    if repo not in _LOADED:
        _release()
        print(f"{tag} loading MLX model {repo} (first use; downloads on cache miss).")
        _LOADED[repo] = load(resolve(repo, tag))
    return _LOADED[repo]
