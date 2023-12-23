import logging

import numpy as np
import torch

import earth2mip.grid
from earth2mip.crps import crps_from_empirical_cdf

logger = logging.getLogger(__name__)


def weighted_average(x, w, dim):
    return torch.mean(x * w, dim) / torch.mean(w, dim)


def area_average(grid: earth2mip.grid.LatLonGrid, x: torch.Tensor):
    lat = torch.tensor(grid.lat, device=x.device)[:, None]
    cos_lat = torch.cos(torch.deg2rad(lat))
    return weighted_average(x, cos_lat, dim=[-2, -1])


def score(
    grid: earth2mip.grid.LatLonGrid,
    ensemble: dict[int, torch.Tensor],
    obs: torch.Tensor,
    device: torch.device = torch.device("cuda"),
) -> dict[str, np.ndarray]:
    """Set of standardized scores for lagged ensembles

    Includes:
    - crps
    - mse of ensemble mean
    - variance about ensemble mean
    - MSE of the first ensemble member

    Args:
        ensemble: mapping of lag to (c, ...)
        obs: (c, ...)
        device: device where the computation should be performed. defaults to
            cuda.

    Returns:
        dict of str to (channel,) shaped metrics numpy arrays.

    """
    # need to run this after since pandas.Index doesn't support cupy
    out = {}
    ens = torch.stack(list(ensemble.values()), dim=0)
    ensemble_dim = 0

    # compute all the metrics
    obs, ens = obs.to(device), ens.to(device)
    crps = crps_from_empirical_cdf(obs, ens)
    out["crps"] = area_average(grid, crps)

    num = (ens.mean(ensemble_dim) - obs) ** 2
    out["MSE_mean"] = area_average(grid, num)

    num = ens.var(ensemble_dim, unbiased=True)
    out["variance"] = area_average(grid, num)

    if 0 in ensemble:
        i = list(ensemble.keys()).index(0)
        num = (ens[i] - obs) ** 2
        out["MSE_det"] = area_average(grid, num)

    return {k: v.cpu().numpy().ravel() for k, v in out.items()}
