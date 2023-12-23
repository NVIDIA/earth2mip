import numpy as np
import torch

from earth2mip.grid import equiangular_lat_lon_grid
from earth2mip.lagged_ensembles.score import score


def test_score(regtest):
    # Create test data
    grid = equiangular_lat_lon_grid(12, 24)

    ngrid = np.prod(grid.shape)
    nchannel = 10
    ntot = ngrid * nchannel
    arr = torch.arange(ntot).reshape(1, nchannel, *grid.shape) / ntot
    ensemble = {
        0: arr - 1,
        1: arr,
        -1: arr + 1,
    }
    obs = torch.zeros_like(ensemble[0][0])

    # Call the score function
    result = score(grid, ensemble, obs)

    # Assert the output shape
    for key in sorted(result):
        print(key, file=regtest)
        with np.printoptions(precision=3):
            print(result[key], file=regtest)
