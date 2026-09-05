"""Keep small numerical regression tests deterministic and inexpensive."""

import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_torch():
    torch.set_num_threads(2)
    torch.manual_seed(42)
