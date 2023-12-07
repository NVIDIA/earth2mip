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

import datetime
import hashlib
import pathlib
import subprocess
from test.initial_conditions.test_hdf5 import create_hdf5

import numpy as np
import pytest
import torch
import xarray

import earth2mip.forecast_metrics_io
import earth2mip.grid
import earth2mip.networks.dlwp
from earth2mip import (
    inference_ensemble,
    inference_medium_range,
    schema,
    score_ensemble_outputs,
    weather_events,
)
from earth2mip._channel_stds import channel_stds
from earth2mip.inference_ensemble import run_basic_inference
from earth2mip.networks import get_model, persistence


def run(args):
    return subprocess.check_call(["coverage", "run", *args])  # noqa: S603 S607


def checksum_reduce_precision(arr, digits=3):
    most_significant = max(arr.max(), -arr.min())
    least = most_significant / 10**digits
    arr = (most_significant / least).astype(np.int32)
    checksum = hashlib.md5(arr.data)  # noqa: S324
    return checksum.hexdigest()


class get_data_source:
    def __init__(self, inference):
        self.channel_names = inference.in_channel_names
        self.grid = inference.grid
        arr = np.ones([len(inference.in_channel_names), *inference.grid.shape])
        self.arr = arr
        self.channel_names = inference.out_channel_names

    def __getitem__(self, time) -> np.ndarray:
        return self.arr


def test_inference_ensemble(tmp_path):
    inference = persistence(package=None)
    data_source = get_data_source(inference)
    time = datetime.datetime(2018, 1, 1)
    channel_stds["a"] = 0.0
    channel_stds["b"] = 0.0
    channel_stds["c"] = 0.0
    config = schema.EnsembleRun(
        weather_model="dummy",
        simulation_length=8,
        output_path=tmp_path.as_posix(),
        weather_event=schema.WeatherEvent(
            properties=weather_events.WeatherEventProperties(
                name="test", start_time=time
            ),
            domains=[
                weather_events.Window(
                    name="globe",
                    diagnostics=[
                        weather_events.Diagnostic(
                            type="raw",
                            channels=inference.out_channel_names,
                        )
                    ],
                )
            ],
        ),
    )

    inference_ensemble.run_inference(
        inference, config, data_source=data_source, progress=True
    )

    path = tmp_path / "ensemble_out_0.nc"
    ds = xarray.open_dataset(path.as_posix(), decode_times=False)
    assert ds.time[0].item() == 0
    out = tmp_path / "out"
    score_ensemble_outputs.main(tmp_path.as_posix(), out.as_posix(), score=False)


def test_checksum_reduce_precision(regtest):
    # Test case 1: Basic example
    arr1 = np.array([1.23456, 2.34567, 3.45678])
    arr2 = np.array([1, 2, 3])
    assert checksum_reduce_precision(arr1, digits=1) == checksum_reduce_precision(
        arr2, digits=1
    )

    arr1 = np.array([0.23456, 2.34567, 3.45678])
    arr2 = np.array([1, 2, 3])
    assert checksum_reduce_precision(arr1, digits=1) != checksum_reduce_precision(
        arr2, digits=2
    )

    print(checksum_reduce_precision(arr1), file=regtest)


def test_inference_medium_range(tmpdir, regtest):
    if not torch.cuda.is_available():
        pytest.skip("need gpu")
    inference = persistence(package=None).cuda()
    data_source = get_data_source(inference)
    time = datetime.datetime(2018, 1, 1)
    dt = datetime.timedelta(hours=6)
    times = [time + dt * k for k in range(10)]
    mean = np.zeros((len(inference.out_channel_names), 721, 1440))
    metrics = inference_medium_range.score_deterministic(
        inference, n=5, initial_times=times, time_mean=mean, data_source=data_source
    )
    metrics["acc"].attrs["checksum"] = checksum_reduce_precision(metrics.acc, digits=3)
    metrics["rmse"].attrs["checksum"] = checksum_reduce_precision(
        metrics.rmse, digits=3
    )
    metrics.info(regtest)


@pytest.mark.slow
@pytest.mark.parametrize("url", ["e2mip://fcn", "e2mip://dlwp"])
def test_run_basic_inference(url):
    time_loop = get_model(url, device="cuda:0")
    data_source = get_data_source(time_loop)
    ds = run_basic_inference(
        time_loop, n=4, data_source=data_source, time=datetime.datetime(2018, 1, 1)
    )
    print(ds)


@pytest.mark.cli
def test_inference_medium_range_cli(tmp_path: pathlib.Path):
    create_hdf5(
        tmp_path,
        2018,
        20,
        grid=earth2mip.grid.equiangular_lat_lon_grid(721, 1440),
        channels=earth2mip.networks.dlwp.CHANNELS,
    )
    output_path = tmp_path / "out"
    run(
        [
            "-m",
            "earth2mip.inference_medium_range",
            "--data",
            tmp_path.as_posix(),
            "-n",
            "5",
            "--start-time",
            "2018-01-02",
            "--end-time",
            "2018-01-02",
            "e2mip://dlwp",
            output_path.as_posix(),
        ]
    )
    series = earth2mip.forecast_metrics_io.read_metrics(output_path.as_posix())
    assert not series.empty


@pytest.mark.cli
def test_lagged_ensemble_cli(tmp_path: pathlib.Path):
    create_hdf5(
        tmp_path,
        2018,
        40,
        grid=earth2mip.grid.equiangular_lat_lon_grid(721, 1440),
        channels=earth2mip.networks.dlwp.CHANNELS,
    )
    output_path = tmp_path / "out"
    run(
        [
            "earth2mip/lagged_ensembles/__main__.py",
            "--data",
            tmp_path.as_posix(),
            "--start-time",
            "2018-01-02",
            "--end-time",
            "2018-01-04",
            "--lags",
            "1",
            "--leads",
            "3",
            "--model",
            "e2mip://dlwp",
            "--output",
            output_path.as_posix(),
        ]
    )
    series = earth2mip.forecast_metrics_io.read_metrics(output_path.as_posix())
    assert not series.empty
