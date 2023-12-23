import numpy as np
import pytest
import torch

from earth2mip.grid import equiangular_lat_lon_grid
from earth2mip.lagged_ensembles.score import score


def _run_score(nlat, nlon, nchannel, device="cpu"):
    # Create test data
    grid = equiangular_lat_lon_grid(nlat, nlon)

    ngrid = np.prod(grid.shape)
    ntot = ngrid * nchannel
    arr = torch.arange(ntot).reshape(1, nchannel, *grid.shape) / ntot
    arr = arr.to(device)
    ensemble = {
        0: arr - 1,
        1: arr,
        -1: arr + 1,
    }
    obs = torch.zeros_like(ensemble[0][0]).to(device)

    # Call the score function
    return score(grid, ensemble, obs)


def test_score_regression(regtest):
    # Create test data
    result = _run_score(12, 14, 10)
    # Assert the output shape
    for key in sorted(result):
        print(key, file=regtest)
        with np.printoptions(precision=3):
            print(result[key].numpy(), file=regtest)


@pytest.mark.slow
def test_score_large_data():
    """Make a test to avoid ooms"""
    torch.cuda.reset_peak_memory_stats()
    _run_score(721, 1440, 200, device="cuda")
    print(torch.cuda.memory_summary())
