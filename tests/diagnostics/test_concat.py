import torch
import pytest
import earth2mip.diagnostic as diag
from earth2mip.diagnostic.identity import Identity
from earth2mip.diagnostic.filter import Filter
from earth2mip.diagnostic.concat import Concat
from earth2mip.schema import Grid


@pytest.mark.parametrize("device", ["cuda:0"])
def test_get_concat(device, in_channels=['a','b','c','d'], grid=Grid.grid_721x1440):

    package = Identity.load_package()
    func1 = Identity.load_diagnostic(package, in_channels[:2], grid)

    package = Filter.load_package()
    func2 = Filter.load_diagnostic(package, in_channels[2:], [in_channels[2]], grid)

    model = diag.get_diagnostic('concat', None, [func1, func2])
    input = torch.randn(1, len(in_channels), len(grid.lat), len(grid.lon)).to(device)
    output = model(input)
    
    package = Filter.load_package()
    model2 = Filter.load_diagnostic(package, model.in_channels, model.out_channels, grid)
    target = model2(input)

    assert torch.allclose(target, output)

@pytest.mark.parametrize("device", ["cuda:0"])
def test_load_concat(device, in_channels=['a','b','c','d'], grid=Grid.grid_721x1440):

    package = Filter.load_package()
    func1 = Filter.load_diagnostic(package, in_channels[:2], [in_channels[0]], grid)

    package = Filter.load_package()
    func2 = Filter.load_diagnostic(package, in_channels[2:], [in_channels[2]], grid)

    package = Concat.load_package()
    model = Concat.load_diagnostic(package, [func1, func2])
    input = torch.randn(1, len(in_channels), 1, 1).to(device)
    output = model(input)
    
    package = Filter.load_package()
    model2 = Filter.load_diagnostic(package, model.in_channels, model.out_channels, grid)
    target = model2(input)

    assert torch.allclose(target, output)