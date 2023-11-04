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

from earth2mip.networks import graphcast
import earth2mip.grid

from graphcast.graphcast import TASK, TASK_13_PRECIP_OUT, TASK_13
import torch
import datetime
import numpy as np

import pytest

from earth2mip.model_registry import Package


@pytest.mark.parametrize("task", [TASK, TASK_13_PRECIP_OUT, TASK_13])
def test_graphcast_time_loop(task):
    nlat = 4
    nlon = 8
    ngrid = nlat * nlon
    batch = 1
    history = 2
    grid = earth2mip.grid.equiangular_lat_lon_grid(nlat, nlon)

    def forward(rng, x):
        assert x.shape == (ngrid, batch, len(in_codes))
        return np.zeros([ngrid, batch, len(target_codes)])

    static_variables = {
        "land_sea_mask": np.zeros(grid.shape),
        "geopotential_at_surface": np.zeros(grid.shape),
    }

    in_codes, target_codes = graphcast.get_codes_from_task_config(task)
    mean = np.zeros(len(in_codes))
    scale = np.ones(len(in_codes))
    diff_scale = np.ones(len(target_codes))
    loop = graphcast.GraphcastTimeLoop(
        forward, static_variables, mean, scale, diff_scale, task_config=task, grid=grid
    )
    initial_time = datetime.datetime(2018, 1, 1)

    x = torch.zeros([batch, history, len(loop.in_channel_names), nlat, nlon])
    for k, (time, y, _) in enumerate(loop(initial_time, x)):
        assert not np.all(np.isnan(y.numpy()))
        assert y.shape == (batch, len(loop.out_channel_names), nlat, nlon)
        assert isinstance(y, torch.Tensor)
        if k == 1:
            break


@pytest.mark.slow
def test_load_time_loop():
    root = "gs://dm_graphcast"
    package = Package(root, seperator="/")
    time_loop = graphcast.load_time_loop_operational(package)
    assert isinstance(time_loop, graphcast.GraphcastTimeLoop)
