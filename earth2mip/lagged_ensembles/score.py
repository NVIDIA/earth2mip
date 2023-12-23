import logging

import cupy
import torch
import xarray

import earth2mip.grid
from earth2mip.xarray import metrics

logger = logging.getLogger(__name__)

use_cupy = True
if use_cupy:
    import cupy as np
else:
    import numpy as np


def score(grid: earth2mip.grid.LatLonGrid, ensemble, obs: np.ndarray):
    """
    Args:
        ensemble: mapping of lag to (c, ...)
        obs: (c, ...)

    Returns:gg
        (c,)
    """
    import dask

    dask.config.set(scheduler="single-threaded")
    obs = xarray.DataArray(data=np.asarray(obs), dims=["channel", "lat", "lon"])
    # need to run this after since pandas.Index doesn't support cupy
    lat = xarray.DataArray(dims=["lat"], data=np.asarray(grid.lat))

    out = {}
    ens = torch.stack(list(ensemble.values()), dim=0)
    ensemble_xr = xarray.DataArray(
        data=np.asarray(ens), dims=["ensemble", "time", *obs.dims]
    )
    ensemble_xr = ensemble_xr.chunk(lat=32)
    obs = obs.chunk(lat=32)
    # need to chunk to avoid OOMs
    with metrics.properscoring_with_cupy():
        out = metrics.score_ensemble(
            ensemble_xr, obs, lat=lat, ensemble_keys=list(ensemble)
        )

    mempool = cupy.get_default_memory_pool()
    logger.debug(
        "bytes used: %0.1f\ttotal: %0.1f",
        mempool.used_bytes() / 2**30,
        mempool.total_bytes() / 2**30,
    )
    out = {k: v.data for k, v in out.items()}
    return out
