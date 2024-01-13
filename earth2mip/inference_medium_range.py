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
import datetime
import logging
import os
import tempfile
from typing import IO, List

import numpy as np
import pandas as pd
import torch
import xarray as xr
from modulus.distributed.manager import DistributedManager

import earth2mip.forecast_metrics_io
from earth2mip import _cli_utils, config, initial_conditions, time_loop
from earth2mip.initial_conditions import hdf5

__all__ = ["score_deterministic"]


class RMSE:
    output_names = ["mse"]

    def __init__(self, weight=None):
        self._xy = {}
        self.weight = weight

    def _mean(self, x):
        if self.weight is not None:
            x = self.weight * x
            denom = self.weight.mean(-1).mean(-1)
        else:
            denom = 1

        num = x.mean(0).mean(-1).mean(-1)
        return num / denom

    def call(self, truth, pred):
        xy = self._mean((truth - pred) ** 2)
        return (xy.cpu(),)

    def gather(self, seq):
        return torch.sqrt(sum(seq) / len(seq))


class ACC:
    output_names = ["xx", "yy", "xy"]

    def __init__(self, mean, weight=None):
        self.mean = mean
        self._xy = {}
        self._xx = {}
        self._yy = {}
        self.weight = weight

    def _mean(self, x):
        if self.weight is not None:
            x = self.weight * x
            denom = self.weight.mean(-1).mean(-1)
        else:
            denom = 1

        num = x.mean(0).mean(-1).mean(-1)
        return num / denom

    def call(self, truth, pred):
        xx = self._mean((truth - self.mean) ** 2).cpu()
        yy = self._mean((pred - self.mean) ** 2).cpu()
        xy = self._mean((pred - self.mean) * (truth - self.mean)).cpu()
        return xx, yy, xy

    def gather(self, seq):
        """seq is an iterable of (xx, yy, xy) tuples"""
        # transpose seq
        xx, yy, xy = zip(*seq)

        xx = sum(xx)
        xy = sum(xy)
        yy = sum(yy)
        return xy / torch.sqrt(xx) / torch.sqrt(yy)


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("inference")


def flat_map(func, seq, *args):
    for x in seq:
        yield from func(x, *args)


def run_forecast(
    model: time_loop.TimeLoop,
    n,
    initial_times,
    device,
    data_source: initial_conditions.base.DataSource,
    mean,
    f: IO[str],
):
    mean = mean.squeeze()
    assert mean.ndim == 3  # noqa

    nlat = len(model.grid.lat)
    channels = [
        data_source.channel_names.index(name) for name in model.out_channel_names
    ]
    mean = mean[channels, :nlat]
    mean = torch.from_numpy(mean).to(device)

    lat = np.deg2rad(model.grid.lat)
    assert lat.ndim == 1  # noqa
    weight = np.cos(lat)[:, np.newaxis]
    weight_torch = torch.from_numpy(weight).to(device)

    acc = ACC(mean, weight=weight_torch)
    metrics = [acc, RMSE(weight=weight_torch)]

    def process(initial_time):
        logger.info(f"Running {initial_time}")
        x = initial_conditions.get_initial_condition_for_model(
            time_loop=model, data_source=data_source, time=initial_time
        )
        logger.debug("Initial Condition Loaded.")
        i = -1
        for valid_time, data, _ in model(x=x, time=initial_time):
            assert data.shape[1] == len(model.out_channel_names)  # noqa
            i += 1
            if i > n:
                break

            lead_time = valid_time - initial_time
            logger.debug(f"{valid_time}")
            verification_torch = initial_conditions.get_data_from_source(
                data_source=data_source,
                time=valid_time,
                channel_names=model.out_channel_names,
                grid=model.grid,
                n_history_levels=1,
                device=model.device,
            )
            # select first history level
            verification_torch = verification_torch[:, -1]
            for metric in metrics:
                outputs = metric.call(verification_torch, data)
                for name, tensor in zip(metric.output_names, outputs):
                    v = tensor.cpu().numpy()
                    for c_idx in range(len(model.out_channel_names)):
                        earth2mip.forecast_metrics_io.write_metric(
                            f,
                            initial_time,
                            lead_time,
                            model.out_channel_names[c_idx],
                            name,
                            value=v[c_idx],
                        )

    for initial_time in initial_times:
        process(initial_time)


def score_deterministic(
    model: time_loop.TimeLoop, n: int, initial_times, data_source, time_mean
) -> xr.Dataset:
    """Compute deterministic accs and rmses

    Args:
        model: the inference class
        n: the number of lead times
        initial_times: the initial_times to compute over
        data_source: a mapping from time to dataset, used for the initial
            condition and the scoring
        time_mean: a (channel, lat, lon) numpy array containing the time_mean.
            Used for ACC.

    Returns:
        metrics: an xarray dataset wtih this structure::
            netcdf dlwp.baseline {
            dimensions:
                    lead_time = 57 ;
                    channel = 7 ;
                    initial_time = 1 ;
            variables:
                    int64 lead_time(lead_time) ;
                            lead_time:units = "hours" ;
                    string channel(channel) ;
                    double acc(lead_time, channel) ;
                            acc:_FillValue = NaN ;
                    double rmse(lead_time, channel) ;
                            rmse:_FillValue = NaN ;
                    int64 initial_times(initial_time) ;
                            initial_times:units = "days since 2018-11-30 12:00:00" ;
                            initial_times:calendar = "proleptic_gregorian" ;
            }
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = f"cuda:{rank % world_size}"
    else:
        rank = 0
        world_size = 1
        device = "cuda:0"

    with tempfile.TemporaryDirectory() as tmpdir:
        save_scores(
            model,
            n,
            initial_times,
            data_source,
            time_mean,
            output_directory=tmpdir,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        series = earth2mip.forecast_metrics_io.read_metrics(tmpdir)
        return time_average_metrics(series)


def time_average_metrics(series: pd.Series) -> xr.Dataset:
    """Average the metrics across initial time and compute ACC, RMSE

    Note, this contrasts from other uses of ACC like weather bench 2.0, since
    the ACC is only formed after the means are taken.
    """
    data_array = series.to_xarray()
    dataset = data_array.to_dataset(dim="metric")
    mean = dataset.mean("initial_time")
    out = xr.Dataset()
    out["rmse"] = np.sqrt(mean["mse"])
    out["acc"] = mean["xy"] / np.sqrt(mean["xx"] * mean["yy"])
    out["initial_times"] = dataset["initial_time"]
    return out


def save_scores(
    model: time_loop.TimeLoop,
    n: int,
    initial_times: List[datetime.datetime],
    data_source: initial_conditions.base.DataSource,
    time_mean: np.ndarray,
    output_directory: str,
    rank: int = 0,
    world_size: int = 1,
    device: str = "cuda",
) -> None:
    """Compute deterministic skill scores, saving the results the a csv file

    Saves the sufficient statistics to compute ACC and RMSE to a csv file for
    each (lead_time, initial_time, channel) tuple.

    For ACC these are xx, xy, and yy. So ACC = E[xy] / sqrt(E[xx] * E[yy]).

    For RMSE this is MSE. So RMSE=sqrt(E[MSE]).

    E is an averaging operator.

    Args:
        model: the inference class
        n: the number of lead times
        initial_times: the initial_times to compute over
        data_source: a mapping from time to dataset, used for the initial
            condition and the scoring
        time_mean: a (channel, lat, lon) numpy array containing the time_mean.
            Used for ACC.

    Returns:
        metrics

    """
    local_initial_times = initial_times[rank::world_size]
    os.makedirs(output_directory, exist_ok=True)
    csv_path = os.path.join(output_directory, f"{rank}.csv")
    with open(csv_path, "a") as f:
        run_forecast(
            model,
            n=n,
            device=device,
            initial_times=local_initial_times,
            data_source=data_source,
            mean=time_mean,
            f=f,
        )


def main():
    parser = argparse.ArgumentParser()
    _cli_utils.add_model_args(parser, required=True)
    _cli_utils.TimeRange.add_args(parser.add_argument_group("Initial Time Selection"))
    parser.add_argument("output")
    parser.add_argument("-n", type=int, default=4)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="shard index. Often set to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=1,
        help="number of shards. Often set to SLURM_ARRAY_TASK_COUNT.",
    )
    # TODO refactor this to a shared place
    parser.add_argument(
        "--data", type=str, help="path to hdf5 root directory containing data.json"
    )

    args = parser.parse_args()
    DistributedManager.initialize()
    dist = DistributedManager()
    initial_times = _cli_utils.TimeRange.from_args(args)

    if args.shard >= args.n_shards:
        raise ValueError("shard must be less than n-shards")

    if args.test:
        initial_times = initial_times[-dist.world_size :]

    model = _cli_utils.model_from_args(args, dist.device)

    data_source = hdf5.DataSource.from_path(
        args.data or config.ERA5_HDF5_73, channel_names=model.in_channel_names
    )
    # time mean
    save_scores(
        model,
        n=args.n,
        initial_times=initial_times,
        data_source=data_source,
        time_mean=data_source.time_means,
        output_directory=args.output,
        rank=args.shard * args.n_shards + dist.rank,
        world_size=dist.world_size * args.n_shards,
        device=dist.device,
    )


if __name__ == "__main__":
    main()
