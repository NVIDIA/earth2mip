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

import dataclasses
import datetime
import json
from typing import List

import numpy as np
import xarray
from modulus.utils import filesystem

import earth2mip.grid
from earth2mip.datasets.era5 import METADATA
from earth2mip.initial_conditions import base


def _get_filename(time: datetime.datetime, lead_time: str):
    date_format = f"%Y%m%d/%Hz/0p4-beta/oper/%Y%m%d%H%M%S-{lead_time}-oper-fc.grib2"
    return time.strftime(date_format)


def _get_channel(c: str, **kwargs) -> xarray.DataArray:
    """

    Parameters:
    -----------
    c: channel id
    **kwargs: variables in ecmwf data
    """
    # handle 2d inputs
    if c in kwargs:
        return kwargs[c]
    else:
        varcode, pressure_level = c[0], int(c[1:])
        return kwargs[varcode].interp(isobaricInhPa=pressure_level)


def get(time: datetime.datetime, channels: List[str]):
    root = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/"
    path = root + _get_filename(time, "0h")
    local_path = filesystem._download_cached(path)
    dataset_0h = xarray.open_dataset(local_path, engine="cfgrib")

    # get t2m and other things from 12 hour forecast initialized 12 hours before
    # The HRES is only initialized every 12 hours
    path = root + _get_filename(time - datetime.timedelta(hours=12), "12h")
    local_path = filesystem._download_cached(path)
    forecast_12h = xarray.open_dataset(local_path, engine="cfgrib")

    channel_data = [
        _get_channel(
            c,
            u10m=dataset_0h.u10,
            v10m=dataset_0h.v10,
            u100m=dataset_0h.u10,
            v100m=dataset_0h.v10,
            sp=dataset_0h.sp,
            t2m=forecast_12h.t2m,
            msl=forecast_12h.msl,
            tcwv=forecast_12h.tciwv,
            t=dataset_0h.t,
            u=dataset_0h.u,
            v=dataset_0h.v,
            r=dataset_0h.r,
            z=dataset_0h.gh * 9.81,
        )
        for c in channels
    ]

    array = np.stack([d for d in channel_data], axis=0)
    darray = xarray.DataArray(
        array,
        dims=["channel", "lat", "lon"],
        coords={
            "channel": channels,
            "lon": dataset_0h.longitude.values,
            "lat": dataset_0h.latitude.values,
            "time": time,
        },
    )
    return darray


@dataclasses.dataclass
class DataSource(base.DataSource):
    def __init__(self, channel_names: List[str]):
        self._channel_names = channel_names

    @property
    def channel_names(self) -> List[str]:
        return self._channel_names

    @property
    def grid(self) -> earth2mip.grid.LatLonGrid:
        return earth2mip.grid.equiangular_lat_lon_grid(721, 1440)

    def __getitem__(self, time: datetime.datetime) -> np.ndarray:
        ds = get(time, self.channel_names)
        ds = ds.expand_dims("time", axis=0)
        # move to earth2mip.channels

        # TODO refactor interpolation to another place
        metadata = json.loads(METADATA.read_text())
        lat = np.array(metadata["coords"]["lat"])
        lon = np.array(metadata["coords"]["lon"])
        ds = ds.roll(lon=len(ds.lon) // 2, roll_coords=True)
        ds["lon"] = ds.lon.where(ds.lon >= 0, ds.lon + 360)
        assert min(ds.lon) >= 0, min(ds.lon)  # noqa
        return ds.interp(lat=lat, lon=lon, kwargs={"fill_value": "extrapolate"})
