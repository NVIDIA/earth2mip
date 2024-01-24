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
import logging
import os
import pathlib
import shutil
import warnings
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import requests
import xarray
from modulus.utils.filesystem import LOCAL_CACHE
from tqdm import tqdm

import earth2mip.grid
from earth2mip.initial_conditions import base

logger = logging.getLogger(__name__)

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
    gfs_channels: List[str],
) -> np.ndarray:
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

    # Make temp grib folder
    pathlib.Path(GFS_CACHE).mkdir(parents=True, exist_ok=True)
    # Get index file
    gfs_chunks = get_gfs_chunks(time_gfs)

    # Loop through channels and download grib of each
    logger.info(f"Downloading {len(gfs_channels)} grib files:")
    for idname in tqdm(gfs_channels):
        get_gfs_grib_file(time_gfs, gfs_chunks, idname, f"{GFS_CACHE}/{idname}.grb")

    # Convert gribs to xarray dataset
    data = np.empty((len(gfs_channels), 721, 1440))
    logger.info(f"Processing {len(gfs_channels)} grib files:")
    for i, name in enumerate(tqdm(gfs_channels)):
        ds = xarray.open_dataset(f"{GFS_CACHE}/{name}.grb", engine="cfgrib")

        for v in ds:
            field = ds[v]
            break

        # If geopotential height multiply by gravity to get geopotential
        if name.startswith("HGT"):
            field = field * 9.81
        data[i] = field

    # Clean up
    shutil.rmtree(GFS_CACHE)

    return data


def _get_gfs_name_dict() -> Dict[str, str]:
    """

    Returns:
        out: out[channel] = gfs_channel


    """

    gfs_channels = [
        "UGRD:10 m above ground",
        "VGRD:10 m above ground",
        "UGRD:100 m above ground",
        "VGRD:100 m above ground",
        "TMP:2 m above ground",
        "PRES:surface",
        "PRMSL:",
        "PWAT:entire atmosphere",
        "UGRD:50 mb",
        "UGRD:100 mb",
        "UGRD:150 mb",
        "UGRD:200 mb",
        "UGRD:250 mb",
        "UGRD:300 mb",
        "UGRD:400 mb",
        "UGRD:500 mb",
        "UGRD:600 mb",
        "UGRD:700 mb",
        "UGRD:850 mb",
        "UGRD:925 mb",
        "UGRD:1000 mb",
        "VGRD:50 mb",
        "VGRD:100 mb",
        "VGRD:150 mb",
        "VGRD:200 mb",
        "VGRD:250 mb",
        "VGRD:300 mb",
        "VGRD:400 mb",
        "VGRD:500 mb",
        "VGRD:600 mb",
        "VGRD:700 mb",
        "VGRD:850 mb",
        "VGRD:925 mb",
        "VGRD:1000 mb",
        "HGT:50 mb",
        "HGT:100 mb",
        "HGT:150 mb",
        "HGT:200 mb",
        "HGT:250 mb",
        "HGT:300 mb",
        "HGT:400 mb",
        "HGT:500 mb",
        "HGT:600 mb",
        "HGT:700 mb",
        "HGT:850 mb",
        "HGT:925 mb",
        "HGT:1000 mb",
        "TMP:50 mb",
        "TMP:100 mb",
        "TMP:150 mb",
        "TMP:200 mb",
        "TMP:250 mb",
        "TMP:300 mb",
        "TMP:400 mb",
        "TMP:500 mb",
        "TMP:600 mb",
        "TMP:700 mb",
        "TMP:850 mb",
        "TMP:925 mb",
        "TMP:1000 mb",
        "RH:50 mb",
        "RH:100 mb",
        "RH:150 mb",
        "RH:200 mb",
        "RH:250 mb",
        "RH:300 mb",
        "RH:400 mb",
        "RH:500 mb",
        "RH:600 mb",
        "RH:700 mb",
        "RH:850 mb",
        "RH:925 mb",
        "RH:1000 mb",
        "SPFH:50 mb",
        "SPFH:100 mb",
        "SPFH:150 mb",
        "SPFH:200 mb",
        "SPFH:250 mb",
        "SPFH:300 mb",
        "SPFH:400 mb",
        "SPFH:500 mb",
        "SPFH:600 mb",
        "SPFH:700 mb",
        "SPFH:850 mb",
        "SPFH:925 mb",
        "SPFH:1000 mb",
    ]

    channels = [
        "u10m",
        "v10m",
        "u100m",
        "v100m",
        "t2m",
        "sp",
        "msl",
        "tcwv",
        "u50",
        "u100",
        "u150",
        "u200",
        "u250",
        "u300",
        "u400",
        "u500",
        "u600",
        "u700",
        "u850",
        "u925",
        "u1000",
        "v50",
        "v100",
        "v150",
        "v200",
        "v250",
        "v300",
        "v400",
        "v500",
        "v600",
        "v700",
        "v850",
        "v925",
        "v1000",
        "z50",
        "z100",
        "z150",
        "z200",
        "z250",
        "z300",
        "z400",
        "z500",
        "z600",
        "z700",
        "z850",
        "z925",
        "z1000",
        "t50",
        "t100",
        "t150",
        "t200",
        "t250",
        "t300",
        "t400",
        "t500",
        "t600",
        "t700",
        "t850",
        "t925",
        "t1000",
        "r50",
        "r100",
        "r150",
        "r200",
        "r250",
        "r300",
        "r400",
        "r500",
        "r600",
        "r700",
        "r850",
        "r925",
        "r1000",
        "q50",
        "q100",
        "q150",
        "q200",
        "q250",
        "q300",
        "q400",
        "q500",
        "q600",
        "q700",
        "q850",
        "q925",
        "q1000",
    ]
    return dict(zip(channels, gfs_channels))


class DataSource(base.DataSource):
    def __init__(self, channels: List[str]) -> None:
        lookup = _get_gfs_name_dict()
        self._gfs_channels = [lookup[c] for c in channels]
        self._channel_names = channels

    @property
    def grid(self) -> earth2mip.grid.LatLonGrid:
        return earth2mip.grid.equiangular_lat_lon_grid(721, 1440)

    @property
    def channel_names(self) -> List[str]:
        return self._channel_names

    def __getitem__(self, time: datetime.datetime) -> np.ndarray:
        return get(time, self._gfs_channels)
