import torch
from earth2mip.diagnostic.wind_gust import WindGust

def test_wind_gust():

    package = WindGust.load_package()
    model = WindGust.load_diagnostic(package, device="cuda:0")

    x = torch.randn(1, len(model.in_channels), len(model.in_grid.lat), len(model.in_grid.lon)).to("cuda:0")

    out = model(x)

    assert out.size() == (1, len(model.out_channels), len(model.out_grid.lat), len(model.out_grid.lon))