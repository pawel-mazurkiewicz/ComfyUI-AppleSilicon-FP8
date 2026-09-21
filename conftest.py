import os

# Before torch loads: 2.14 reads this once at init, so the per-test setdefault in
# the int8 reference comparisons comes too late there and eager _int_mm raises.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pytest
import torch

collect_ignore = ["__init__.py"]

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an MPS (Apple Silicon) device",
)
