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
from typing import List
import datetime
import dataclasses
from earth2mip import schema
import xarray
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from cdsapi import Client

import logging

logging.getLogger("cdsapi").setLevel(logging.WARNING)
import urllib3

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)  # Hack to disable SSL warnings


@dataclasses.dataclass
class DataSource:
    channel_names: List[str]

    @property
    def time_means(self):
        raise NotImplementedError()

    def __getitem__(self, time: datetime.datetime):
        return _get_channels(time, self.channel_names)


# CDS mapping from channel names to CDS names, tuple means pressure level data
def create_CDS_channel_mapping():
    channel_mapping = {
        "u10m": "10m_u_component_of_wind",
        "v10m": "10m_v_component_of_wind",
        "t2m": "2m_temperature",
        "sp": "surface_pressure",
        "msl": "mean_sea_level_pressure",
        "tcwv": "total_column_water_vapour",
        "u100m": "100m_u_component_of_wind",
        "v100m": "100m_v_component_of_wind",
    }

    # for variables depending on pressure level
    _pressure_levels = [
        1,
        2,
        3,
        5,
        7,
        10,
        20,
        30,
        50,
        70,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        775,
        800,
        825,
        850,
        875,
        900,
        925,
        950,
        975,
        1000,
    ]

    # add the pressure level variables
    for level in _pressure_levels:
        # add u component of wind
        channel = f"u{level}"
        cds_name = "u_component_of_wind"
        channel_mapping[channel] = (cds_name, level)

        # add v component of wind
        channel = f"v{level}"
        cds_name = "v_component_of_wind"
        channel_mapping[channel] = (cds_name, level)

        # add geopotential
        channel = f"z{level}"
        cds_name = "geopotential"
        channel_mapping[channel] = (cds_name, level)

        # add temperature
        channel = f"t{level}"
        cds_name = "temperature"
        channel_mapping[channel] = (cds_name, level)

        # add relative humidity
        channel = f"r{level}"
        cds_name = "relative_humidity"
        channel_mapping[channel] = (cds_name, level)

        # add specific humidity
        channel = f"q{level}"
        cds_name = "specific_humidity"
        channel_mapping[channel] = (cds_name, level)
    return channel_mapping


def retrieve_channel_data(client, cds_variable, time, area, grid, format, file):
    if isinstance(cds_variable, tuple):  # it's a pressure level variable
        cds_name, level = cds_variable
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": [cds_name],
                "pressure_level": [level],
                "year": time.strftime("%Y"),
                "month": time.strftime("%m"),
                "day": time.strftime("%d"),
                "time": time.strftime("%H:%M"),
                "area": area,
                "grid": grid,
                "format": format,
            },
            file,
        )
    elif isinstance(cds_variable, str):  # it's a single level variable
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [cds_variable],
                "year": time.strftime("%Y"),
                "month": time.strftime("%m"),
                "day": time.strftime("%d"),
                "time": time.strftime("%H:%M"),
                "area": area,
                "grid": grid,
                "format": format,
            },
            file,
        )
    else:
        raise NotImplementedError(cds_variable)


def get(time: datetime.datetime, channel_set: schema.ChannelSet):
    warnings.warn(
        DeprecationWarning("Will be removed. Please use CDSDataSource instead.")
    )
    channels = channel_set.list_channels()
    return _get_channels(time, channels)


def _download_channels_legacy(client, channels, time):
    grid = (0.25, 0.25)
    area = (90, -180, -90, 180)
    format = "netcdf"

    channel_mapping = create_CDS_channel_mapping()

    # create a list of arguments for each call to retrieve_channel_data
    cds_variables = [channel_mapping[c] for c in channels]

    def get(cds_variable):
        tmp_file = tempfile.mktemp(suffix=".nc")
        try:
            retrieve_channel_data(
                client, cds_variable, time, area, grid, format, tmp_file
            )
            # return first array in dataset
            ds = xarray.open_dataset(tmp_file)
            for v in ds:
                return ds[v].load()
        finally:
            os.unlink(tmp_file)

    with ThreadPoolExecutor() as executor:
        data_arrays = executor.map(get, cds_variables)

    darray = xarray.Dataset(
        {channel: arr for channel, arr in zip(channels, data_arrays)}
    )
    return darray


def _get_channels(time: datetime.datetime, channels: List[str]):
    client = Client(progress=False, quiet=True)

    ds = _download_channels_legacy(client, channels, time)

    # Concatenate channels
    darray = ds.to_array(dim="channel")
    darray = darray.transpose("time", "channel", "latitude", "longitude")

    # Rename lat/lon
    darray = darray.rename({"latitude": "lat", "longitude": "lon"})
    darray = darray.assign_coords(lon=darray["lon"] + 180.0)
    darray = darray.roll(lon=1440 // 2)

    return darray
