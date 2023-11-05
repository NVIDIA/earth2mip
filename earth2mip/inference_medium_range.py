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

from typing import IO, List
import logging

import os
import tempfile
import xarray as xr
import torch
import argparse
import numpy as np
import datetime
from earth2mip import config
from earth2mip import time_loop, initial_conditions
from earth2mip.initial_conditions import hdf5
from earth2mip import _cli_utils
import earth2mip.forecast_metrics_io
from modulus.distributed.manager import DistributedManager


__all__ = ["score_deterministic"]


def get_times(start_time: datetime, end_time: datetime):
    # the IFS data Jaideep downloaded only has 668 steps (up to end of november 2018)
    times = []
    time = start_time
    while time <= end_time:
        times.append(time)
        time += datetime.timedelta(hours=12)
    return times


class RMSE:
    outputs = ["mse"]

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
    outputs = ["xx", "yy", "xy"]

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
    assert mean.ndim == 3

    nlat = len(model.grid.lat)
    channels = [
        data_source.channel_names.index(name) for name in model.out_channel_names
    ]
    mean = mean[channels, :nlat]
    mean = torch.from_numpy(mean).to(device)

    lat = np.deg2rad(model.grid.lat)
    assert lat.ndim == 1
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
            assert data.shape[1] == len(model.out_channel_names)
            i += 1
            if i > n:
                break

            lead_time = valid_time - initial_time
            logger.debug(f"{valid_time}")
            # TODO make this more performant grabs all history steps unnecessarily
            verification_torch = initial_conditions.get_initial_condition_for_model(
                time_loop=model, data_source=data_source, time=valid_time
            )
            # select first history level
            verification_torch = verification_torch[:, 0]
            for metric in metrics:
                outputs = metric.call(verification_torch, data)
                for name, tensor in zip(metric.outputs, outputs):
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


def gather(seq, metrics, model_name, channels):
    outputs_by_lead_time = {}
    initial_times = set()
    for (initial_time, lead_time), metric_values in seq:
        forecasts_at_lead_time = outputs_by_lead_time.setdefault(lead_time, [])
        forecasts_at_lead_time.append(metric_values)
        initial_times.add(initial_time)

    def to_dataset(metric, name):
        outputs = {
            k: [v[name] for v in snapshots]
            for k, snapshots in outputs_by_lead_time.items()
        }
        times, accs = zip(*outputs.items())
        times = list(times)
        acc_arr = [metric.gather(acc) for acc in accs]
        stacked = torch.stack(acc_arr, 0)
        stacked = stacked.cpu().numpy()
        return xr.DataArray(
            stacked,
            dims=["lead_time", "channel"],
            coords={"lead_time": times, "channel": channels},
        ).to_dataset(name=name)

    ds = xr.merge(to_dataset(metric, name) for name, metric in metrics.items())
    ds = ds.assign(
        initial_times=xr.DataArray(list(initial_times), dims=["initial_time"])
    )

    return ds


def score_deterministic(
    model: time_loop.TimeLoop, n: int, initial_times, data_source, time_mean
):
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
        metrics::
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
    parser.add_argument("output")
    parser.add_argument("-n", type=int, default=4)
    parser.add_argument("--test", action="store_true")
    # TODO refactor this to a shared place
    parser.add_argument(
        "--data", type=str, help="path to hdf5 root directory containing data.json"
    )
    parser.add_argument("--start-time", type=str, default="2018-01-01")
    parser.add_argument(
        "--end-time", type=str, default="2018-12-01", help="final time (inclusive)."
    )

    args = parser.parse_args()
    DistributedManager.initialize()
    dist = DistributedManager()

    start_time = datetime.datetime.fromisoformat(args.start_time)
    end_time = datetime.datetime.fromisoformat(args.end_time)
    initial_times = get_times(start_time, end_time)
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
        rank=dist.rank,
        world_size=dist.world_size,
        device=dist.device,
    )


if __name__ == "__main__":
    main()
