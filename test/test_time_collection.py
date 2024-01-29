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
import json
import os
import subprocess

import numpy as np
import pytest
import torch
import xarray

import earth2mip.grid
import earth2mip.schema
from earth2mip.datasets.hindcast import open_forecast
from earth2mip.time_collection import run_over_initial_times

DIR = os.path.dirname(__file__)
os.environ["PYTHONPATH"] = DIR + ":" + os.getenv("PYTHONPATH", ":")


def create_model_package(tmp_path):
    package_dir = tmp_path / "mock_package"
    package_dir.mkdir()

    # Create metadata.json
    metadata = {
        "architecture_entrypoint": "mock_plugin:load",
        "n_history": 0,
        "grid": "721x1440",
        "in_channels": list(range(73)),
        "out_channels": list(range(73)),
    }
    with open(package_dir.joinpath("metadata.json"), "w") as f:
        json.dump(metadata, f)

    # Create numpy arrays
    global_means = np.zeros((1, 73, 1, 1))
    global_stds = np.ones((1, 73, 1, 1))
    np.save(package_dir.joinpath("global_means.npy"), global_means)
    np.save(package_dir.joinpath("global_stds.npy"), global_stds)

    return package_dir


@pytest.mark.slow
@pytest.mark.xfail
def test_time_collection(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs gpu and data")

    torch.init_process_group()
    model_package = create_model_package(tmp_path)

    config = os.path.join(DIR, "configs/medium-test.json")
    model = f"file://{model_package.as_posix()}"

    root = str(tmp_path / "test")
    subprocess.check_call(
        ["python3", "-m", "earth2mip.make_job", model, config, root]  # noqa: S603 S607
    )  # noqa: S603 S607
    subprocess.check_call(  # noqa: S603 S607
        [  # noqa: S603 S607
            "torchrun",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(min([torch.cuda.device_count(), 2])),
            "-m",
            "earth2mip.time_collection",
            root,
        ]
    )  # noqa: S603 S607

    ds = open_forecast(root, group="mean.zarr")
    assert isinstance(ds, xarray.Dataset)


@pytest.mark.slow
def test_run_over_initial_times(tmp_path, regtest):
    class MockTimeLoop:
        in_channel_names = ["b", "a"]
        out_channel_names = ["b", "a"]
        time_step = datetime.timedelta(hours=6)
        history_time_step = datetime.timedelta(hours=6)
        n_history_levels = 1
        grid = earth2mip.grid.equiangular_lat_lon_grid(2, 2)
        device = "cpu"
        dtype = torch.float

        def __call__(self, time, x):
            while True:
                yield time, x[:, 0], None
                time += self.time_step

    class MockData:
        grid = MockTimeLoop.grid
        channel_names = MockTimeLoop.in_channel_names

        def __getitem__(self, time):
            return np.zeros([len(self.channel_names), *self.grid.shape])

    # an template configuration for a short ensemble run
    config_obj = {
        "ensemble_members": 4,
        "simulation_length": 10,
        "weather_event": {
            "properties": {"name": "global", "start_time": "2018-01-01T00:00:00"},
            "domains": [
                {
                    "name": "global",
                    "type": "Window",
                    "diagnostics": [
                        {
                            "type": "raw",
                            "channels": MockTimeLoop.out_channel_names,
                        }
                    ],
                }
            ],
        },
        "output_path": "dummy/output",
        "output_frequency": 1,
        "weather_model": "unused",
        "seed": 12345,
        "use_cuda_graphs": False,
        "ensemble_batch_size": 1,
        "autocast_fp16": False,
        "perturbation_strategy": "correlated",
        "noise_amplitude": 0.05,
        "noise_reddening": 2,
    }

    time_loop = MockTimeLoop()
    data_source = MockData()
    initial_times = [datetime.datetime(2018, 1, 1), datetime.datetime(2018, 1, 2)]
    config = earth2mip.schema.EnsembleRun.parse_obj(config_obj)
    output_path = tmp_path.as_posix()

    run_over_initial_times(
        time_loop=time_loop,
        data_source=data_source,
        initial_times=initial_times,
        config=config,
        output_path=output_path,
        n_post_processing_workers=1,
    )

    # open the data and ensure its format is the same as the regression data
    ds = open_forecast(output_path, "mean.zarr")
    # print the CDL-like schema of the dataset to the regression fixture
    # ``regtest``
    # delete some non-reproducible attrs
    del ds.attrs["date_created"]
    del ds.attrs["history"]
    ds.info(regtest)
