# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import concurrent.futures
import datetime
import logging
from functools import partial
import argparse

import cupy
import pandas as pd
import torch
import xarray

from earth2mip import forecasts, _cli_utils
from earth2mip.initial_conditions.era5 import HDF5DataSource
from earth2mip.datasets.hindcast import open_forecast
from earth2mip.lagged_ensembles import core
from earth2mip.xarray import metrics
from earth2mip.xarray.utils import concat_dict, to_cupy
from earth2mip import config

# patch the proper scoring imports
use_cupy = True
if use_cupy:
    import cupy as np
else:
    import numpy as np


logger = logging.getLogger(__name__)


async def lagged_average_simple(
    *,
    observations,
    run_forecast,
    score,
    lags=2,
    n=10,
):

    scores = {}
    async for (j, l), ensemble, obs in core.yield_lagged_ensembles(
        observations=observations,
        forecast=run_forecast,
        lags=lags,
        n=n,
    ):
        scores.setdefault(j, {})[l] = score(ensemble, obs)
    return scores


def get_times_2018(nt):
    times = [
        datetime.datetime(2018, 1, 1) + k * datetime.timedelta(hours=12)
        for k in range(nt)
    ]
    return times


class Observations:
    def __init__(self, times, pool, data_source, device=None):
        self.pool = pool
        self.device = device
        self.times = times
        self.data_source = data_source

    def _get_time(self, time):
        return self.data_source[time]

    async def __getitem__(self, i):
        """
        Returns (channel, lat, lon)
        """
        time = self.times[i]
        logger.debug("Loading %s", time)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.pool, self._get_time, time)

    def __len__(self):
        return len(self.times)


def score(channel_names, ensemble, obs):
    """
    Args:
        ensemble: list of (c, ...)
        obs: (c, ...)

    Returns:gg
        (c,)
    """
    import dask

    dask.config.set(scheduler="single-threaded")
    obs = to_cupy(obs.drop(["time", "channel"])).assign_coords(
        time=obs.time, channel=obs.channel
    )
    lat = to_cupy(obs.lat)

    out = {}
    ens = torch.stack(list(ensemble.values()), dim=0)
    coords = {**obs.coords}
    coords["channel"] = channel_names
    ensemble_xr = xarray.DataArray(
        np.asarray(ens), dims=["ensemble", *obs.dims], coords=coords
    )
    # add ensemble dimension
    # the convention is that ensemble member 0 is the deterministic (i.e. best)
    # one
    ensemble_xr = ensemble_xr.assign_coords(
        ensemble=xarray.Variable(["ensemble"], list(ensemble))
    )

    ensemble_xr = ensemble_xr.chunk(lat=32)
    obs = obs.chunk(lat=32)
    # need to chunk to avoid OOMs
    pred_align, obs_align = xarray.align(ensemble_xr, obs)
    with metrics.properscoring_with_cupy():
        out = metrics.score_ensemble(pred_align, obs_align, lat=lat)

    mempool = cupy.get_default_memory_pool()
    logger.debug(
        "bytes used: %0.1f\ttotal: %0.1f",
        mempool.used_bytes() / 2**30,
        mempool.total_bytes() / 2**30,
    )
    return out


def collect_score(score, times) -> pd.DataFrame:
    """traverse the collected scores and collate into a data frame

    score[j][l][series] is a DataArray of `series` for valid index `j` and lead
    time `l`

    """

    # save data with these columns
    # time,valid_time,model,series,t850,u10m,v10m,t2m,z500,initial_time
    dt = times[1] - times[0]

    flat = {}
    for j in score:
        for ell in score[j]:
            for series in score[j][ell]:
                arr = score[j][ell][series]
                arr = arr.copy()
                try:
                    # is a cupy array
                    arr.data = arr.data.get()
                except AttributeError:
                    # otherwise do nothing
                    pass
                arr = arr.squeeze()
                flat[(times[j] - ell * dt, ell * dt, series)] = arr

    # idx = pd.MultiIndex.from_tuples(list(flat.keys()), names=['initial_time', 'time'])
    combined = concat_dict(flat, key_names=["initial_time", "time", "series"])
    df = combined.to_dataset(dim="channel").to_dataframe().reset_index()
    df["valid_time"] = df["initial_time"] + df["time"]
    del df["time"]
    del df["key"]
    return df


def main(args):
    """Run a lagged ensemble scoring

    Can be run against either a fcn model (--model), a forecast directory as
    output by earth2mip.time_collection (--forecast_dir), persistence forecast
    (--persistence), or deterministic IFS (--ifs).

    Saves data as csv files (1 per rank).

    Examples:

        torchrun --nproc_per_node 2 --nnodes 1 -m earth2mip.lagged_ensembles --model sfno_73ch --inits 10 --leads 5 --lags 4

    """  # noqa

    times = list(get_times_2018(args.inits))
    FIELDS = ["u10m", "v10m", "z500", "t2m", "t850"]
    pool = concurrent.futures.ThreadPoolExecutor()

    data_source = HDF5DataSource.from_path(args.data or config.ERA5_HDF5_73)
    obs = Observations(times=times, pool=pool, data_source=data_source, device="cpu")

    try:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    except ValueError:
        pass

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    device = torch.device("cuda", rank % torch.cuda.device_count())
    if args.model:
        timeloop = _cli_utils.model_from_args(args, device=device)
        run_forecast = forecasts.TimeLoopForecast(
            timeloop, times=times, observations=obs
        )
    elif args.forecast_dir:
        run_forecast = forecasts.XarrayForecast(
            open_forecast(args.forecast_dir, group="mean.zarr"),
            times=times,
            fields=FIELDS,
        )
    elif args.ifs:
        # TODO fix this import error
        # TODO convert ifs to zarr so we don't need custom code
        from earth2mip.datasets.deterministic_ifs import open_deterministic_ifs

        run_forecast = forecasts.XarrayForecast(open_deterministic_ifs(args.ifs))
    elif args.persistence:
        run_forecast = forecasts.Persistence
    else:
        raise ValueError(
            "need to provide one of --persistence --ifs --forecast-dir or --model."
        )

    if rank == 0:
        logging.basicConfig(level=logging.INFO)

    scores_future = lagged_average_simple(
        observations=obs,
        score=partial(score, run_forecast.channel_names),
        run_forecast=run_forecast,
        lags=args.lags,
        n=args.leads,
    )

    with torch.cuda.device(device):
        scores = asyncio.run(scores_future)
    df = collect_score(scores, times)
    path = f"{args.output}.{rank:03d}.csv"
    print(f"saving scores to {path}")
    # remove headers from other ranks so it is easy to cat the files
    df.to_csv(path, header=(rank == 0))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Your CLI description here", usage=main.__doc__
    )

    parser.add_argument("--data", type=str, help="Path to data file")
    _cli_utils.add_model_args(parser, required=False)
    parser.add_argument("--forecast_dir", type=str, help="Path to forecast directory")
    parser.add_argument("--ifs", type=str, default="", help="IFS parameter")
    parser.add_argument("--persistence", action="store_true", help="Enable persistence")
    parser.add_argument("--inits", type=int, default=10, help="Number of inits")
    parser.add_argument("--lags", type=int, default=4, help="Number of lags")
    parser.add_argument("--leads", type=int, default=54, help="Number of leads")
    parser.add_argument("--output", type=str, default=".", help="Output directory")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
