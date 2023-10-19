import torch
import pytest
from earth2mip.diagnostic.climate_net import ClimateNet


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cuda:0"])
def test_climate_net(device):

    package = ClimateNet.load_package()
    model = ClimateNet.load_diagnostic(package, device)

    x = torch.randn(
        1, len(model.in_channels), len(model.in_grid.lat), len(model.in_grid.lon)
    ).to(device)
    out = model(x)
    assert out.size() == (
        1,
        len(model.out_channels),
        len(model.out_grid.lat),
        len(model.out_grid.lon),
    )
