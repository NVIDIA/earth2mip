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

from graphcast.graphcast import TASK, TASK_13_PRECIP_OUT, TASK_13
from earth2mip.initial_conditions import cds
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
    lat = np.linspace(90, -90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)

    def forward(rng, x):
        assert x.shape == (ngrid, batch, len(in_codes))
        return np.zeros([ngrid, batch, len(target_codes)])

    static_variables = {
        "land_sea_mask": np.zeros((nlat, nlon)),
        "geopotential_at_surface": np.zeros((nlat, nlon)),
    }

    in_codes, target_codes = graphcast.get_codes_from_task_config(task)
    mean = np.zeros(len(in_codes))
    scale = np.ones(len(in_codes))
    diff_scale = np.ones(len(target_codes))
    loop = graphcast.GraphcastTimeLoop(
        forward,
        static_variables,
        mean,
        scale,
        diff_scale,
        task_config=task,
        lat=lat,
        lon=lon,
    )
    time = datetime.datetime(2018, 1, 1)

    x = torch.zeros([batch, history, len(loop.in_channel_names), nlat, nlon])
    for k, (time, y, _) in enumerate(loop(time, x)):
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


@pytest.mark.slow
def test_tisr_matches_cds():
    import matplotlib.pyplot as plt

    time = datetime.datetime(2018, 1, 1)
    ds = cds.DataSource(channel_names=["tisr"])
    data = ds[time]
    fig, (a, b, c) = plt.subplots(nrows=3)
    tisr = graphcast.toa_incident_solar_radiation(time, data.lat, data.lon) / 3600

    tisr_era5 = data.sel(channel="tisr")[0] / 3600

    vmax = 1500
    print(tisr_era5 / tisr)

    im = a.pcolormesh(tisr, vmin=0, vmax=vmax)
    b.pcolormesh(tisr_era5, vmin=0, vmax=vmax)
    plt.colorbar(im, ax=[a, b])
    im = c.pcolormesh(tisr - tisr_era5)

    n = tisr - tisr_era5
    rmse = np.sqrt(np.mean(n * n)).item()
    c.set_title(f"RMSE (W/m2): {rmse:.02f}")
    plt.colorbar(im, ax=c)

    fig.savefig("out.png")

    plt.figure()
    i = 300
    plt.plot(data.lon, tisr_era5[i], label="era5")
    plt.plot(data.lon, tisr[i], label="computed")
    plt.legend()
    plt.savefig("line.png")

    return tisr_era5, tisr
