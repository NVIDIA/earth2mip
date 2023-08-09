import torch
import pytest
from earth2mip import regrid
from earth2mip import schema


def test_get_regridder():
    src = schema.Grid.grid_721x1440
    dest = schema.Grid.s2s_challenge
    try:
        f = regrid.get_regridder(src, dest)
    except FileNotFoundError as e:
        pytest.skip(f"{e}")
    x = torch.ones(1, 1, 721, 1440)
    y = f(x)
    assert y.shape == (1, 1, 121, 240)
    assert torch.allclose(y, torch.ones_like(y))
