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
import jax
import jax.numpy
import numpy as np
import pytest
from graphcast import xarray_jax

from earth2mip.model_registry import Package
from earth2mip.networks import graphcast
from earth2mip.networks.graphcast import get_channel_names, get_forcings
from earth2mip.time_loop import TimeStepperLoop


def test_get_forcings():
    # time.shape == (1, 1) == (batch, time)
    time = np.array([[np.datetime64("2018-01-01T00:00:00")]])
    lat = np.arange(-90, 90)
    lon = np.arange(0, 360)
    f = get_forcings(time, lat, lon)
    w = np.cos(np.deg2rad(f.lat))
    ans = f["toa_incident_solar_radiation"].weighted(w).mean().item()
    seconds_in_hour = 3600
    solar_constant = 1361 / 4
    approx = solar_constant * seconds_in_hour
    assert ans == pytest.approx(approx, rel=0.05)
    assert f["toa_incident_solar_radiation"].shape == (1, 1, 180, 360)
    assert f["day_progress_cos"].dims == ("batch", "time", "lon")
    assert f["day_progress_sin"].dims == ("batch", "time", "lon")
    assert f["year_progress_cos"].dims == ("batch", "time")
    assert f["year_progress_sin"].dims == ("batch", "time")


def test_get_forcings_jax():
    time = np.array([[np.datetime64("2018-01-01T00:00:00")]])
    lat = jax.numpy.arange(-90, 90)
    lon = jax.numpy.arange(0, 360)
    inputs = get_forcings(time, lat, lon)
    for v in inputs:
        v = xarray_jax.jax_data(inputs[v])
        assert v.device() == lat.device()


def test_get_channel_names():
    names = get_channel_names(
        [
            "toa_incidient_solar_radiation",
            "2m_temperature",
            "geopotential",
            "specific_humidity",
        ],
        pressure_levels=[1, 2, 3],
    )
    # no tisr since it is a forcing variable
    assert names == ["z1", "z2", "z3", "q1", "q2", "q3", "t2m"]


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_load_time_loop():
    root = "gs://dm_graphcast"
    package = Package(root, seperator="/")
    time_loop = graphcast.load_time_loop_operational(package)
    assert isinstance(time_loop, TimeStepperLoop)
