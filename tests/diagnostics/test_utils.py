import pytest
import torch
from earth2mip.diagnostic.utils import filter_channels


@pytest.mark.parametrize("device", ["cuda:0"])
def test_filter_channels(device):

    input = torch.randn(1, 1, 3, 2, 2).to(device)
    output = filter_channels(input, ["a", "b", "c"], ["a", "b"])
    assert torch.allclose(input[:, :, :2], output)

    input = torch.randn(3, 5, 5).to(device)
    output = filter_channels(input, ["a", "b", "c"], ["c"])
    assert torch.allclose(input[2:], output)
