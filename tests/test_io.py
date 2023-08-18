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

from earth2mip import netcdf
import numpy as np
import netCDF4 as nc
from earth2mip.schema import Grid
from earth2mip.weather_events import Window, Diagnostic
import torch


def test_initialize_netcdf(tmp_path):
    domain = Window(
        name="TestAverage",
        lat_min=-15,
        lat_max=15,
        diagnostics=[Diagnostic(type="raw", function="", channels=["tcwv"])],
    )
    lat = np.array([-20, 0, 20])
    lon = np.array([0, 1, 2])
    n_ensemble = 1
    path = tmp_path / "a.nc"
    with nc.Dataset(path.as_posix(), "w") as ncfile:
        netcdf.initialize_netcdf(
            ncfile,
            [domain],
            Grid("720x1440"),
            lat,
            lon,
            n_ensemble,
            torch.device(type="cpu"),
        )
