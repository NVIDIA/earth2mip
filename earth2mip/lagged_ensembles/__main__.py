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

import argparse
import asyncio
import concurrent.futures
import datetime
import logging
import os
from typing import List

# patch the proper scoring imports
import torch

import earth2mip.forecast_metrics_io
import earth2mip.grid
from earth2mip import _cli_utils, config, forecasts
from earth2mip.datasets.hindcast import open_forecast
from earth2mip.initial_conditions import get_data_from_source, hdf5
from earth2mip.lagged_ensembles import core, score

logger = logging.getLogger(__name__)


async def lagged_average_simple(
    *,
    observations,
    run_forecast,
    lags=2,
    n=10,
    times: List[datetime.datetime],
    time_step: datetime.timedelta,
    filename: str,
):
    async for (j, k), ensemble, obs in core.yield_lagged_ensembles(
        observations=observations,
        forecast=run_forecast,
        min_lag=-lags,
        max_lag=lags,
        n=n,
    ):
        initial_time = times[j] - k * time_step
        lead_time = time_step * k

        ensemble_cuda = {lag: x.cuda() for lag, x in ensemble.items()}
        out = score.score(run_forecast.grid, ensemble_cuda, obs.cuda())

        with open(filename, "a") as f:
            earth2mip.forecast_metrics_io.write_metric(
                f,
                initial_time=initial_time,
                lead_time=lead_time,
                channel="",
                metric="ensemble_size",
                value=len(ensemble),
            )
            for metric_name, darray in out.items():
                assert darray.shape == (len(run_forecast.channel_names),)  # noqa
                for i in range(len(run_forecast.channel_names)):
                    earth2mip.forecast_metrics_io.write_metric(
                        f,
                        initial_time=initial_time,
                        lead_time=lead_time,
                        channel=run_forecast.channel_names[i],
                        metric=metric_name,
                        value=darray[i].item(),
                    )
        logger.info(f"finished with {initial_time} {lead_time}")


class Observations:
    def __init__(
        self,
        times,
        pool,
        data_source,
        channel_names,
        grid: earth2mip.grid.LatLonGrid,
        device: torch.device = torch.device("cpu"),
    ):
        self.pool = pool
        self.device = device
        self.times = times
        self.data_source = data_source
        self.channel_names = channel_names
        self._grid = grid

    def _get_time(self, time):
        return get_data_from_source(
            self.data_source,
            time,
            self.channel_names,
            self._grid,
            n_history_levels=1,
            device=self.device,
        )

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


def main(args):
    """Run a lagged ensemble scoring

    Can be run against either a fcn model (--model), a forecast directory as
    output by earth2mip.time_collection (--forecast_dir), persistence forecast
    (--persistence), or deterministic IFS (--ifs).

    Saves data as csv files (1 per rank).

    Examples:

        torchrun --nproc_per_node 2 --nnodes 1 -m earth2mip.lagged_ensembles --model sfno_73ch --inits 10 --leads 5 --lags 4

    """  # noqa

    times = _cli_utils.TimeRange.from_args(args)
    FIELDS = ["u10m", "v10m", "z500", "t2m", "t850"]
    pool = concurrent.futures.ThreadPoolExecutor()

    data_source = hdf5.DataSource.from_path(args.data or config.ERA5_HDF5)

    try:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    except ValueError:
        pass

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    device = torch.device("cuda", rank % torch.cuda.device_count())

    if rank == 0:
        logging.basicConfig(level=logging.INFO)

    if args.model:
        timeloop = _cli_utils.model_from_args(args, device=device)
        run_forecast = forecasts.TimeLoopForecast(
            timeloop, times=times, data_source=data_source
        )
    elif args.forecast_dir:
        run_forecast = forecasts.XarrayForecast(
            open_forecast(args.forecast_dir, group="mean.zarr"),
            times=times,
            fields=FIELDS,
            device=device,
        )
    elif args.ifs:
        # TODO fix this import error
        # TODO convert ifs to zarr so we don't need custom code
        from earth2mip.datasets.deterministic_ifs import open_deterministic_ifs

        run_forecast = forecasts.XarrayForecast(
            open_deterministic_ifs(args.ifs), device=device
        )
    elif args.persistence:
        run_forecast = forecasts.Persistence
    else:
        raise ValueError(
            "need to provide one of --persistence --ifs --forecast-dir or --model."
        )

    if args.channels:
        run_forecast = forecasts.select_channels(
            run_forecast, channel_names=args.channels.split(",")
        )

    logger.info(
        f"number of timesteps: {len(times)}, "
        f"start time: {times[0]}, end_time: {times[-1]}"
    )

    obs = Observations(
        times=times,
        pool=pool,
        data_source=data_source,
        device="cpu",
        grid=run_forecast.grid,
        channel_names=run_forecast.channel_names,
    )

    logging.basicConfig(level=logging.INFO)

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, f"{rank:03d}.csv")
    print(f"saving scores to {output_path}")
    with torch.cuda.device(device), torch.no_grad():
        scores_future = lagged_average_simple(
            observations=obs,
            run_forecast=run_forecast,
            lags=args.lags,
            n=args.leads,
            filename=output_path,
            times=times,
            time_step=times[1] - times[0],
        )
        asyncio.run(scores_future)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Your CLI description here", usage=main.__doc__
    )

    parser.add_argument("--data", type=str, help="Path to data file")
    _cli_utils.add_model_args(parser, required=False)
    _cli_utils.TimeRange.add_args(parser)
    parser.add_argument("--forecast_dir", type=str, help="Path to forecast directory")
    parser.add_argument("--ifs", type=str, default="", help="IFS parameter")
    parser.add_argument("--persistence", action="store_true", help="Enable persistence")
    parser.add_argument("--lags", type=int, default=4, help="Number of lags")
    parser.add_argument("--leads", type=int, default=54, help="Number of leads")
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--channels",
        type=str,
        default="",
        help="Channels to score, comma seperated list."
        "Example: 't2m,t850,z500,u10m,v10m'."
        "Defaults to all channels in the forecast.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
