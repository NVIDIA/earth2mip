# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

from earth2mip import forecasts
import xarray
import numpy as np
import datetime
import torch
import pytest


class MockTimeLoop:
    in_channel_names = ["b", "a"]
    out_channel_names = ["b", "a"]
    time_step = datetime.timedelta(hours=6)

    def __call__(self, time, x):
        assert torch.all(x == torch.tensor([1, 0], device=x.device))
        while True:
            yield time, x, None
            time += self.time_step


async def getarr():
    arr = np.arange(3)
    coords = {}
    coords["channel"] = ["a", "b", "c"]
    return xarray.DataArray(arr, dims=["channel"], coords=coords)


async def test_TimeLoopForecast():
    if not torch.cuda.is_available():
        pytest.skip("No Cuda")

    times = [
        datetime.datetime(1, 1, 1) + datetime.timedelta(hours=12) * k for k in range(3)
    ]
    mock_obs = [getarr() for t in times]

    forecast = forecasts.TimeLoopForecast(MockTimeLoop(), times, mock_obs)

    iter = forecast[0]
    k = 0
    async for state in iter:
        k += 1
        if k >= 4:
            break
