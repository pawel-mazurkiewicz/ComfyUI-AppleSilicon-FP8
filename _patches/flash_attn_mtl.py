"""Wire mtlflashattn into ComfyUI: a guarded `flash_attn` drop-in plus a rerouted
F.scaled_dot_product_attention on MPS.

Both integrations are independently guarded and never fatal; anything outside the SDPA
gate, and any kernel error, falls back to stock SDPA.

  MTLFLASHATTN_SDPA=off    disable the SDPA patch   (legacy: APPLESILICON_FP8_SDPA=off)
  MTLFLASHATTN_SHIM=off    disable the flash_attn shim
  MTLFLASHATTN_SDPA_MIN_SEQ / _FAST_MIN_SEQ / _MIN_GB   gate thresholds
"""
from __future__ import annotations

import os

TAG = "[AppleSilicon-FP8/flash]"

# the old node's env knobs, kept working
_LEGACY_ENV = {
    "APPLESILICON_FP8_SDPA": "MTLFLASHATTN_SDPA",
    "APPLESILICON_FP8_SDPA_MIN_GB": "MTLFLASHATTN_SDPA_MIN_GB",
}


def _alias_legacy_env():
    """Mirror the legacy vars onto the new names, unless the new one is already set."""
    for old, new in _LEGACY_ENV.items():
        val = os.environ.get(old)
        if val is not None and new not in os.environ:
            os.environ[new] = val


def install():
    _alias_legacy_env()

    try:
        import torch
    except Exception:
        return  # no torch -> nothing to patch
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return

    try:
        from metal_flash_attn import _shim
        from metal_flash_attn import sdpa as mfa_sdpa
    except Exception:
        print(
            f"{TAG} mtlflashattn not installed -- flash_attn drop-in and fast SDPA "
            f"are off. Install it with:  pip install mtlflashattn",
            flush=True,
        )
        return

    shim_on = False
    try:
        shim_on = _shim.install()
    except Exception as e:
        print(f"{TAG} flash_attn shim failed to install ({e})", flush=True)

    sdpa_on = False
    try:
        sdpa_on = mfa_sdpa.install()
    except Exception as e:
        print(f"{TAG} SDPA patch failed to install ({e})", flush=True)

    parts = []
    if shim_on:
        parts.append("flash_attn drop-in active")
    if sdpa_on:
        parts.append(
            "F.scaled_dot_product_attention -> mtlflashattn on MPS "
            f"(correctness>={os.environ.get('MTLFLASHATTN_SDPA_MIN_SEQ', '4096')} tok, "
            f"fast-tier>={os.environ.get('MTLFLASHATTN_SDPA_FAST_MIN_SEQ', '1024')} tok, "
            f"oom>={os.environ.get('MTLFLASHATTN_SDPA_MIN_GB', '12')} GB)"
        )
    if parts:
        print(f"{TAG} {'; '.join(parts)}.", flush=True)
    else:
        # already active (the shim can auto-load via .pth) or killed by env
        print(
            f"{TAG} mtlflashattn present; flash_attn/SDPA already active or "
            f"disabled via env (MTLFLASHATTN_SHIM/MTLFLASHATTN_SDPA).",
            flush=True,
        )
