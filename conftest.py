import os

# before the torch import: torch 2.14 reads it once at init
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pytest
import torch

collect_ignore = ["__init__.py"]

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an MPS (Apple Silicon) device",
)
