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

import pathlib

import netCDF4 as nc
import pytest
import torch

import earth2mip.grid
from earth2mip import netcdf, weather_events


@pytest.mark.parametrize("cls", ["raw"])
def test_diagnostic(cls: str, tmp_path: pathlib.Path):
    domain = weather_events.Window(
        name="Test",
        lat_min=-15,
        lat_max=15,
        diagnostics=[
            weather_events.Diagnostic(type=cls, function="", channels=["tcwv"])
        ],
    )
    n_ensemble = 2
    path = tmp_path / "a.nc"
    with nc.Dataset(path.as_posix(), "w") as ncfile:
        total_diagnostics = netcdf.initialize_netcdf(
            ncfile,
            [domain],
            earth2mip.grid.equiangular_lat_lon_grid(
                720, 1440, includes_south_pole=False
            ),
            n_ensemble,
            torch.device(type="cpu"),
        )[0]

        for diagnostic in total_diagnostics:
            print(ncfile)
            print(ncfile["Test"])
            nlat = len(ncfile["Test"]["lat"][:])
            nlon = len(ncfile["Test"]["lon"][:])
            data = torch.randn((n_ensemble, 1, nlat, nlon))
            time_index = 0
            batch_id = 0
            batch_size = n_ensemble
            diagnostic.update(data, time_index, batch_id, batch_size)

        if cls == "skill":
            assert "tcwv" in ncfile["Test"]["skill"].variables
        elif cls == "raw":
            assert "tcwv" in ncfile["Test"].variables
        else:
            assert "tcwv" in ncfile["Test"][cls].variables
