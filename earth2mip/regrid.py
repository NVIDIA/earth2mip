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

import netCDF4 as nc
import einops
import torch
import pathlib
import xarray
import numpy as np
from earth2mip.schema import Grid
from earth2mip._config import Settings


class TempestRegridder(torch.nn.Module):
    def __init__(self, file_path):
        super().__init__()
        dataset = nc.Dataset(file_path)
        self.lat = dataset["latc_b"][:]
        self.lon = dataset["lonc_b"][:]

        i = dataset["row"][:] - 1
        j = dataset["col"][:] - 1
        M = dataset["S"][:]

        i = i.data
        j = j.data
        M = M.data

        self.M = torch.sparse_coo_tensor((i, j), M, [max(i) + 1, max(j) + 1]).float()

    def to(self, device):
        self.M = self.M.to(device)
        return self

    def forward(self, x):
        xr = einops.rearrange(x, "b c x y -> b c (x y)")
        yr = xr @ self.M.T
        y = einops.rearrange(
            yr, "b c (x y) -> b c x y", x=self.lat.size, y=self.lon.size
        )
        return y


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


class DropSouthPole(torch.nn.Module):
    def forward(self, x):
        return x[..., :-1, :]


def _get_tempest_regridder(src: Grid, dest: Grid) -> TempestRegridder:
    # TODO map data needs to be available for S2S scoring
    config = Settings()

    # TODO add instructions for how to make the tempest map file
    map_file = (
        pathlib.Path(config.MAP_FILES) / src.value / dest.value / "tempest_map.nc"
    )
    return TempestRegridder(map_file.as_posix())


def get_regridder(src: Grid, dest: Grid):
    if src == dest:
        return Identity()
    elif (src, dest) == (Grid.grid_721x1440, Grid.grid_720x1440):
        return DropSouthPole()
    else:
        return _get_tempest_regridder(src, dest)
    raise NotImplementedError()


def xarray_regrid(src: xarray.Dataset, dest: Grid):
    """Simple function for regridding an xarray dataset to a grid.
    The xarray dataset must have lat and lon coords

    Parameters
    ----------
    src : xr.Dataset
        Input xarray data set
    dest : Grid
        Target grid scheme
    """
    # TODO refactor this into a regridder
    # Subsample / interpolate lat lon grid
    if np.isin(dest.lat, src.coords["lat"].values).all():
        src = src.sel(lat=dest.lat)
    else:
        src = src.interp(lat=dest.lat)

    if np.isin(dest.lon, src.coords["lon"].values).all():
        src = src.sel(lon=dest.lon)
    else:
        src = src.interp(lon=dest.lon)

    return src
