import torch
import earth2mip.diagnostic as e2diag


# @pytest.mark.parametrize("nt", [16, 20])
def test_wind_gust_config():
    # cfg = e2diag.load_config_type('windgust')()
    package = e2diag.get_package("windgust")
    model = e2diag.get_diagnostic("windgust", package, device="cuda:0")
    # model = cfg.initialize()

    x = torch.randn(
        1, len(model.in_channels), len(model.in_grid.lat), len(model.in_grid.lon)
    ).to("cuda:0")
    out = model(x)
    assert out.size() == (
        1,
        len(model.out_channels),
        len(model.out_grid.lat),
        len(model.out_grid.lon),
    )
