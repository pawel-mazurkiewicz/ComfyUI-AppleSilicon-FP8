import pytest
import torch

collect_ignore = ["__init__.py"]

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an MPS (Apple Silicon) device",
)
