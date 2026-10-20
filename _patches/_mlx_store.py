"""Where the mlx_textgen backends keep MLX models: ComfyUI's text_encoders folder, under mlx/."""

import json
import os


def _comfy_dirs():
    try:
        import folder_paths
        return list(folder_paths.get_folder_paths("text_encoders"))
    except Exception:
        return []


def _local_dir(base, repo):
    return os.path.join(base, "mlx", *repo.split("/"))


def _complete(path):
    """config.json plus every weight shard: an interrupted download is resumed, not loaded."""
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    # some repos ship a stale index next to a single model.safetensors
    if os.path.isfile(os.path.join(path, "model.safetensors")):
        return True
    try:
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            shards = set(json.load(f)["weight_map"].values())
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False
    return all(os.path.isfile(os.path.join(path, s)) for s in shards)


def resolve(repo, tag):
    """Local folder for `repo`: as given, text_encoders/mlx/, HF cache, else download there."""
    if os.path.isdir(repo):
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
