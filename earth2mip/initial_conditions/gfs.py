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

import datetime
from earth2mip import schema
from earth2mip.datasets.gfs import METADATA26, METADATA34, METADATA73
from modulus.utils.filesystem import LOCAL_CACHE
import json
import xarray
import numpy as np
import shutil
import pathlib
import os
import requests
import warnings
from dataclasses import dataclass
from typing import List, Union
from tqdm import tqdm

# Max byte check of any one field
# Will error if larger
MAX_BYTE_SIZE = 2000000
# Location to cache grib files
GFS_CACHE = LOCAL_CACHE + "/earth2mip/gfs"


@dataclass
class GFSChunk:
    variable_name: str = "phoo"
    meta_data: str = ""
    start_byte: int = 0
    end_byte: int = 0

    @property
    def byte_range(self) -> int:
        return self.end_byte - self.start_byte

    @property
    def channel_id(self) -> str:
        return ":".join([self.variable_name, self.meta_data])


def _get_index_url(time):
    return (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/"
        + "prod/gfs."
        + time.strftime("%Y%m%d")
        + "/"
        + time.strftime("%H")
        + "/atmos/gfs.t"
        + time.strftime("%H")
        + "z.pgrb2.0p25.f000.idx"
    )


def gfs_available(
    time: datetime.datetime,
) -> bool:
    nearest_hour = time.hour - time.hour % 6
    time_gfs = datetime.datetime(time.year, time.month, time.day, nearest_hour)
    index_url = _get_index_url(time_gfs)
    try:
        r = requests.get(index_url, timeout=5)
        r.raise_for_status()
    except requests.exceptions.RequestException:
        return False
    return True


def get_gfs_chunks(
    time: datetime.datetime,
):
    index_url = _get_index_url(time)
    try:
        r = requests.get(index_url, timeout=5)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        raise SystemExit(e)

    if len(r.text) == 0:
        raise ValueError(f"Empty index file: {r.text}")

    index_lines = r.text.splitlines()
    index_lines = index_lines[:-1]
    output = [GFSChunk()]
    for i, line in enumerate(index_lines):
        lsplit = line.split(":")
        if len(lsplit) < 7:
            continue

        chunk = GFSChunk(
            variable_name=lsplit[3],
            meta_data=lsplit[4],
            start_byte=int(lsplit[1]),
            end_byte=None,
        )
        output.append(chunk)
        # Update previous chunk with end position based on start of current chunk
        output[-2].end_byte = int(lsplit[1]) - 1

        if MAX_BYTE_SIZE < output[-2].byte_range:
            raise ValueError(
                "Byte range in index field found to be too large."
                + f" Parsed byte range {output[-2].byte_range}, max byte"
                + f" range {MAX_BYTE_SIZE}"
            )

    # Pop place holder
    output.pop(0)
    return output


def get_gfs_grib_file(
    time: datetime.datetime,
    gfs_chunks: List[GFSChunk],
    channel_id: str,
    output_file: str,
):
    gfs_url = (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/"
        + "prod/gfs."
        + time.strftime("%Y%m%d")
        + "/"
        + time.strftime("%H")
        + "/atmos/gfs.t"
        + time.strftime("%H")
        + "z.pgrb2.0p25.f000"
    )

    # Get chunk data for this variable
    gfs_chunk = None
    for chunk in gfs_chunks:
        if channel_id in chunk.channel_id:
            gfs_chunk = chunk
            break

    if gfs_chunk is None:
        raise ValueError(f"Variable {channel_id} not found in index")

    start_str = str(gfs_chunk.start_byte) if gfs_chunk.start_byte else "0"
    end_str = str(gfs_chunk.end_byte) if gfs_chunk.end_byte else ""
    headers = {"Range": f"bytes={start_str}-{end_str}"}

    # Send request to GFS
    try:
        with requests.get(gfs_url, headers=headers, stream=True, timeout=10) as r:
            with open(f"{output_file}.tmp", "wb") as f:
                shutil.copyfileobj(r.raw, f)
            r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

    # Finally rename the file
    try:
        os.rename(f"{output_file}.tmp", f"{output_file}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{output_file}.tmp not found in GFS cache. "
            + "its likely failed to download"
        )


def get(
    time: Union[datetime.datetime, None],
    channel_set: schema.ChannelSet,
) -> xarray.DataArray:
    # If no time is provided, use current time
    if time is None:
        time = datetime.datetime.now()
        # Check if most recent time is available, if not fall back 6 hours
        if not gfs_available(time):
            warnings.warn("Closest 6 hour interval not available, falling back 6 hours")
            time = time - datetime.timedelta(hours=6)

    nearest_hour = time.hour - time.hour % 6
    time_gfs = datetime.datetime(time.year, time.month, time.day, nearest_hour)

    if not gfs_available(time_gfs):
        raise ValueError(
            f"Nearest 6 hour time {time_gfs} is not available right now "
            + "(needs to be past 10 days)"
        )

    if channel_set == schema.ChannelSet.var26:
        # move to earth2mip.channels
        metadata = json.loads(METADATA26.read_text())
        channels = metadata["coords"]["channel"]
        gfs_channels = metadata["gfs_coords"]["channel"]
    elif channel_set == schema.ChannelSet.var34:
        # move to earth2mip.channels
        metadata = json.loads(METADATA34.read_text())
        channels = metadata["coords"]["channel"]
        gfs_channels = metadata["gfs_coords"]["channel"]
    elif channel_set == schema.ChannelSet.var73:
        # move to earth2mip.channels
        metadata = json.loads(METADATA73.read_text())
        channels = metadata["coords"]["channel"]
        gfs_channels = metadata["gfs_coords"]["channel"]
    else:
        raise NotImplementedError(channel_set)

    # Make temp grib folder
    pathlib.Path(GFS_CACHE).mkdir(parents=True, exist_ok=True)
    # Get index file
    gfs_chunks = get_gfs_chunks(time_gfs)

    # Loop through channels and download grib of each
    print(f"Downloading {len(channels)} grib files:")
    for idname, outname in zip(tqdm(gfs_channels), channels):
        get_gfs_grib_file(time_gfs, gfs_chunks, idname, f"{GFS_CACHE}/{outname}.grb")

    # Convert gribs to xarray dataset
    data = np.empty((1, len(channels), 721, 1440))
    gfsds = xarray.Dataset(
        {"fields": (["time", "channel", "lat", "lon"], data)},
        coords={
            "time": [time_gfs],
            "channel": metadata["coords"]["channel"],
            "lat": metadata["coords"]["lat"],
            "lon": metadata["coords"]["lon"],
        },
    )

    print(f"Processing {len(channels)} grib files:")
    for i, name in enumerate(tqdm(channels)):
        ds = xarray.open_dataset(f"{GFS_CACHE}/{name}.grb", engine="cfgrib")
        field = ds[list(ds.keys())[0]]
        # If geopotential height multiply by gravity to get geopotential
        if name[0] == "z":
            field = field * 9.81
        gfsds["fields"][0, i] = field

    # Clean up
    shutil.rmtree(GFS_CACHE)

    return gfsds["fields"]
