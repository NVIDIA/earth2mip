import torch
import pytest
import earth2mip.diagnostic as e2diag

@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.parametrize("device", ["cuda:0"])
def test_wind_gust_entrypoint(device):
    # Needs to have internal windgust package installed
    package = e2diag.get_package("windgust")
    model = e2diag.get_diagnostic("windgust", package, device=device)

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
