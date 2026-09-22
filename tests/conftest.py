import os
import sys

import pytest
import torch

# The custom-node dir name contains hyphens, so it isn't an importable package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an MPS (Apple Silicon) device",
)
