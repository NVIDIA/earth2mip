import torch
import pytest
import earth2mip.diagnostic as diag
from earth2mip.diagnostic.identity import Identity
from earth2mip.schema import Grid


@pytest.mark.parametrize("in_channels", [['a','b','c'], ['a','d']])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_get_identity(device, in_channels, grid=Grid.grid_721x1440):

    model = diag.get_diagnostic('identity', None, in_channels, grid)
    input = torch.randn(1, len(in_channels), len(grid.lat), len(grid.lon))
    output = model(input)
    
    assert torch.allclose(input, output)

@pytest.mark.parametrize("in_channels", [['a','b','c'], ['a','d']])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_load_identity(device, in_channels, grid=Grid.grid_721x1440):

    package = Identity.load_package()
    model = Identity.load_diagnostic(package, in_channels, grid)
    input = torch.randn(1, len(in_channels), len(grid.lat), len(grid.lon))
    output = model(input)
    
    assert torch.allclose(input, output)