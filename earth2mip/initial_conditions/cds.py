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

import warnings
import eccodes
from typing import List, Union
import datetime
import dataclasses
from earth2mip import schema
import xarray
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from cdsapi import Client

import logging

logging.getLogger("cdsapi").setLevel(logging.WARNING)
import urllib3

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)  # Hack to disable SSL warnings

# codes database: https://codes.ecmwf.int/grib/param-db/?filter=grib2
CHANNEL_TO_CODE = {
    "z": 129,
    "u": 131,
    "v": 132,
    # w = dp/dt, normally called omega
    "w": 135,
    "t": 130,
    "q": 133,
    "r": 157,
    "t2m": 167,
    "u10m": 165,
    "v10m": 166,
    "u100m": 228246,
    "v100m": 228247,
    "tcwv": 137,
    "sp": 134,
    "msl": 151,
    # total precip
    "tp": 228,
    # total precip accumlated over 6 hours
    "tp06": 260267,
    "tisr": 212,
    "zs": 162051,
    "lsm": 172,
}


def keys_to_vals(d):
    return dict(zip(d.values(), d.keys()))


@dataclasses.dataclass(eq=True, order=True, frozen=True)
class PressureLevelCode:
    id: int
    level: int = 0

    def __str__(self):
        lookup = keys_to_vals(CHANNEL_TO_CODE)
        return lookup[self.id] + str(self.level)


@dataclasses.dataclass(eq=True, order=True, frozen=True)
class SingleLevelCode:
    id: int

    def __str__(self):
        lookup = keys_to_vals(CHANNEL_TO_CODE)
        return lookup[self.id]


def parse_channel(channel: str) -> Union[PressureLevelCode, SingleLevelCode]:
    if channel in CHANNEL_TO_CODE:
        return SingleLevelCode(CHANNEL_TO_CODE[channel])
    else:
        code = CHANNEL_TO_CODE[channel[0]]
        level = int(channel[1:])
        return PressureLevelCode(code, level=int(level))


@dataclasses.dataclass
class DataSource:
    channel_names: List[str]
    client: Client = dataclasses.field(
        default_factory=lambda: Client(progress=False, quiet=False)
    )
    _cache: str = ".cds/"

    @property
    def time_means(self):
        raise NotImplementedError()

    def __getitem__(self, time: datetime.datetime):
        d = os.path.join(self._cache, time.isoformat())
        os.makedirs(d, exist_ok=True)
        return _get_channels(self.client, time, self.channel_names, d)


def get(time: datetime.datetime, channel_set: schema.ChannelSet):
    warnings.warn(
        DeprecationWarning("Will be removed. Please use CDSDataSource instead.")
    )
    channels = channel_set.list_channels()
    ds = DataSource(channels)
    return ds[time]


def _get_cds_requests(codes, time, format):
    grid = (0.25, 0.25)
    area = (90, -180, -90, 180)

    # create a list of arguments for each call to retrieve_channel_data
    levels = set()
    pressure_level_names = set()
    single_level_names = set()
    for v in codes:
        if isinstance(v, PressureLevelCode):  # it's a pressure level variable
            levels.add(v.level)
            pressure_level_names.add(v.id)
        elif isinstance(v, SingleLevelCode):  # it's a single level variable
            single_level_names.add(v.id)

    if pressure_level_names and levels:
        yield (
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": list(pressure_level_names),
                "pressure_level": sorted(levels),
                "year": time.strftime("%Y"),
                "month": time.strftime("%m"),
                "day": time.strftime("%d"),
                "time": time.strftime("%H:%M"),
                "area": area,
                "grid": grid,
                "format": format,
            },
        )

    if single_level_names:
        yield (
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": sorted(single_level_names),
                "year": time.strftime("%Y"),
                "month": time.strftime("%m"),
                "day": time.strftime("%d"),
                "time": time.strftime("%H:%M"),
                "area": area,
                "grid": grid,
                "format": format,
            },
        )


def _parse_files(
    codes: List[Union[SingleLevelCode, PressureLevelCode]], files: List[str]
) -> xarray.DataArray:
    """Retrieve ``codes`` from a list of ``files``

    Returns:
        a data array of all the codes

    """
    arrays = [None] * len(codes)
    for path in files:
        print(path)
        with open(path) as f:
            while True:
                gid = eccodes.codes_grib_new_from_file(f)
                if gid is None:
                    break
                id = eccodes.codes_get(gid, "paramId")
                print(id)
                level = eccodes.codes_get(gid, "level")
                type_of_level = eccodes.codes_get(gid, "typeOfLevel")

                if type_of_level == "surface":
                    code = SingleLevelCode(id)
                else:
                    code = PressureLevelCode(id, level=level)

                nlat = eccodes.codes_get(gid, "Nj")
                nlon = eccodes.codes_get(gid, "Ni")

                lat = eccodes.codes_get_array(gid, "latitudes").reshape(nlat, nlon)
                lon = eccodes.codes_get_array(gid, "longitudes").reshape(nlat, nlon)
                vals = eccodes.codes_get_values(gid).reshape(nlat, nlon)
                eccodes.codes_release(gid)

                try:
                    i = codes.index(code)
                except ValueError:
                    continue

                arrays[i] = vals
    array = np.stack(arrays)
    coords = {}
    coords["lat"] = lat[:, 0]
    coords["lon"] = lon[0, :]
    return xarray.DataArray(array, dims=["channel", "lat", "lon"], coords=coords)


def _download_codes(client, codes, time, d):

    files = []
    format = "grib2"

    def download(arg):
        name, req = arg
        path = os.path.join(d, name + ".grib")
        if not os.path.exists(path):
            client.retrieve(name, req, path)
        return path

    requests = _get_cds_requests(codes, time, format)
    with ThreadPoolExecutor(4) as pool:
        files = pool.map(download, requests)

    darray = _parse_files(codes, files)

    return darray


def _get_channels(client, time: datetime.datetime, channels: List[str], d):
    codes = [parse_channel(c) for c in channels]
    darray = _download_codes(client, codes, time, d)
    return (
        darray.assign_coords(channel=channels)
        .assign_coords(time=time)
        .expand_dims("time")
        .transpose("time", "channel", "lat", "lon")
        .assign_coords(lon=darray["lon"] + 180.0)
        .roll(lon=1440 // 2)
    )
