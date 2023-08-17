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

from earth2mip import config
import xarray
import datetime
from earth2mip import schema
import joblib
import numpy as np
from earth2mip.initial_conditions.era5 import open_era5_xarray, HDF5DataSource
from earth2mip.initial_conditions import ifs
from earth2mip.initial_conditions import cds
from earth2mip.initial_conditions import gfs
from earth2mip.initial_conditions import hrmip

# TODO remove this fcn-mip import
from earth2mip.datasets.era5 import METADATA
import json

__all__ = ["open_era5_xarray", "get", "get_data_source"]


def get_data_source(
    n_history,
    grid,
    channel_set,
    netcdf="",
    initial_condition_source=schema.InitialConditionSource.era5,
):
    if initial_condition_source == schema.InitialConditionSource.era5:
        root = config.get_data_root(channel_set)
        return HDF5DataSource.from_path(root)
    else:
        return LegacyDataSource(
            n_history,
            grid,
            channel_set,
            netcdf=netcdf,
            initial_condition_source=initial_condition_source,
        )


class LegacyDataSource:
    def __init__(
        self,
        n_history,
        grid,
        channel_set,
        netcdf="",
        initial_condition_source=schema.InitialConditionSource.era5,
    ):
        self.n_history = n_history
        self.grid = grid
        self.channel_set = channel_set
        self.initial_condition_source = initial_condition_source
        self.netcdf = ""

    def __getitem__(self, time):
        if self.netcdf:
            return xarray.open_dataset(self.netcdf)["fields"]
        else:
            return ic(
                n_history=self.n_history,
                grid=self.grid,
                time=time,
                channel_set=self.channel_set,
                source=self.initial_condition_source,
            )


def get(
    n_history: int,
    time: datetime.datetime,
    channel_set: schema.ChannelSet,
    source: schema.InitialConditionSource = schema.InitialConditionSource.era5,
) -> xarray.DataArray:
    if source == schema.InitialConditionSource.hrmip:
        ds = hrmip.get(time, channel_set)
        return ds
    elif source == schema.InitialConditionSource.ifs:
        if n_history > 0:
            raise NotImplementedError("IFS initializations only work with n_history=0.")
        ds = ifs.get(time, channel_set)
        ds = ds.expand_dims("time", axis=0)
        # move to earth2mip.channels

        # TODO refactor interpolation to another place
        metadata = json.loads(METADATA.read_text())
        lat = np.array(metadata["coords"]["lat"])
        lon = np.array(metadata["coords"]["lon"])
        ds = ds.roll(lon=len(ds.lon) // 2, roll_coords=True)
        ds["lon"] = ds.lon.where(ds.lon >= 0, ds.lon + 360)
        assert min(ds.lon) >= 0, min(ds.lon)
        return ds.interp(lat=lat, lon=lon, kwargs={"fill_value": "extrapolate"})
    elif source == schema.InitialConditionSource.cds:
        if n_history > 0:
            raise NotImplementedError("CDS initializations only work with n_history=0.")
        ds = cds.get(time, channel_set)
        return ds
    elif source == schema.InitialConditionSource.gfs:
        if n_history > 0:
            raise NotImplementedError("GFS initializations only work with n_history=0.")
        return gfs.get(time, channel_set)
    else:
        raise NotImplementedError(source)


if config.LOCAL_CACHE:
    memory = joblib.Memory(config.LOCAL_CACHE)
    get = memory.cache(get)


def ic(
    time: datetime,
    grid,
    n_history: int,
    channel_set: schema.ChannelSet,
    source: schema.InitialConditionSource,
):
    ds = get(n_history, time, channel_set, source)
    # TODO collect grid logic in one place
    if grid == schema.Grid.grid_720x1440:
        return ds.isel(lat=slice(0, -1))
    elif grid == schema.Grid.grid_721x1440:
        return ds
    else:
        raise NotImplementedError(f"Grid {grid} not supported")
