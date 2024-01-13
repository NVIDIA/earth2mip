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
import test.test_end_to_end

import numpy as np
import pytest
import torch
import xarray

import earth2mip.grid
from earth2mip import forecasts


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


async def test_TimeLoopForecast():
    if not torch.cuda.is_available():
        pytest.skip("No Cuda")

    times = [
        datetime.datetime(1, 1, 1) + datetime.timedelta(hours=12) * k for k in range(3)
    ]

    forecast = forecasts.TimeLoopForecast(
        MockTimeLoop(),
        times,
        data_source=test.test_end_to_end.get_data_source(MockTimeLoop()),
    )

    iter = forecast[0]
    k = 0
    async for state in iter:
        assert state.shape == (1, 2, 2, 2)
        k += 1
        if k >= 4:
            break


async def test_XarrayForecast():

    times = [
        datetime.datetime(1, 1, 1) + datetime.timedelta(hours=12) * k for k in range(3)
    ]

    # setup dataset
    lead_times = [datetime.timedelta(hours=6) * k for k in range(6)]
    grid = earth2mip.grid.equiangular_lat_lon_grid(2, 2)
    channels = ["a", "b"]
    shape = [len(times), len(lead_times), *grid.shape]
    data_vars = {}
    for c in channels:
        data_vars[c] = (("initial_time", "time", "lat", "lon"), np.ones(shape))

    coords = {}
    coords["lat"] = grid.lat
    coords["lon"] = grid.lon
    coords["initial_time"] = times
    coords["time"] = lead_times
    dataset = xarray.Dataset(data_vars=data_vars, coords=coords)

    # wrap it
    forecast = forecasts.XarrayForecast(
        dataset, fields=channels, times=times, device="cpu"
    )

    # test it
    iter = forecast[0]
    k = 0
    async for state in iter:
        assert state.shape == (1, 2, 2, 2)
        k += 1
        if k >= len(times):
            break
