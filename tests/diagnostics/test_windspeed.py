import pytest
import torch
from earth2mip.schema import Grid
from earth2mip.diagnostic import WindSpeed


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("grid", [Grid.grid_721x1440, Grid.grid_720x1440])
def test_wind_speed(device, grid):
    model = WindSpeed.load_diagnostic(None, level="10m", grid=grid)
    x = torch.randn(2, len(model.in_channels), len(grid.lat), len(grid.lon)).to(device)
    out = model(x)
    assert torch.allclose(torch.sqrt(x[:, :1] ** 2 + x[:, 1:] ** 2), out)
