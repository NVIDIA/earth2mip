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

import datetime
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
    return channel_mapping


def retrieve_channel_data(args):
    idx, channel, time, area, grid, format, channel_mapping, client = args
    cds_variable = channel_mapping.get(channel)
    if not cds_variable:
        raise ValueError(f"Unknown channel: {channel}")

    if isinstance(cds_variable, tuple):  # it's a pressure level variable
        cds_name, level = cds_variable
        tmp_file = tempfile.NamedTemporaryFile(suffix=".nc")
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
            tmp_file.name,
        )
    else:  # it's a single level variable
        tmp_file = tempfile.NamedTemporaryFile(suffix=".nc")
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
            tmp_file.name,
        )
    data = xarray.open_dataset(tmp_file.name)
    main_var = list(set(data.variables) - set(data.coords))[
        0
    ]  # main variable is one not in coordinates
    data = data.rename({main_var: channel})  # rename variable to channel
    return idx, data


def retrieve_era5_data(
    channels,
    time,
    file_name,
    area=(90, -180, -90, 180),
    grid=(0.25, 0.25),
    format="netcdf",
):
    c = Client(progress=False, quiet=True)
    channel_mapping = create_CDS_channel_mapping()
    data_arrays = [None] * len(channels)  # pre-allocate list

    # create a list of arguments for each call to retrieve_channel_data
    args = [
        (idx, channel, time, area, grid, format, channel_mapping, c)
        for idx, channel in enumerate(channels)
    ]

    with ThreadPoolExecutor() as executor:
        for idx, data in executor.map(retrieve_channel_data, args):
            data_arrays[idx] = data  # place data into correct position

    combined_data = xarray.merge(data_arrays)
    combined_data.to_netcdf(file_name)


def get(time: datetime.datetime, channel_set: schema.ChannelSet):

    # Get channel data
    channels = channel_set.list_channels()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "file.nc")
        retrieve_era5_data(channels, time, path)
        darray = xarray.open_dataset(path).load()

    # Concatenate channels
    darray = darray.to_array(dim="channel")
    darray = darray.transpose("time", "channel", "latitude", "longitude")

    # Rename lat/lon
    darray = darray.rename({"latitude": "lat", "longitude": "lon"})
    darray = darray.assign_coords(lon=darray["lon"] + 180.0)
    darray = darray.roll(lon=1440 // 2)

    return darray
