import numpy as np
from earth2mip import grid


def test_regular_lat_lon():
    g = grid.regular_lat_lon_grid(721, 1440)
    assert np.all(np.diff(g.lat) == -0.25)
    assert g.shape == (721, 1440)
