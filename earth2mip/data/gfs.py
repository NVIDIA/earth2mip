# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import hashlib
import os
import pathlib
import shutil
from typing import Union

import boto3
import botocore
import numpy as np
import s3fs
import xarray as xr
from botocore import UNSIGNED
from loguru import logger
from modulus.distributed.manager import DistributedManager
from tqdm import tqdm

from earth2mip import config
from earth2mip.lexicon import GFSLexicon


class GFS:
    """The global forecast service (GFS) re-analysis data source provided on an
    equirectangular grid. GFS is a weather forecast model developed by NOAA. This data
    source is provided on a 0.25 degree lat lon grid at 6-hour intervals spanning from
    Feb 26th 2021 to present date.

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True

    Note
    ----
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:
    https://registry.opendata.aws/noaa-gfs-bdp-pds/
    Additional information about GFS solver can be referenced here:
    https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php
    """

    GFS_BUCKET_NAME = "noaa-gfs-bdp-pds"
    MAX_BYTE_SIZE = 2000000

    GFS_LAT = np.linspace(90, -90, 721)
    GFS_LON = np.linspace(0, 359.75, 1440)

    def __init__(self, cache: bool = True):
        self._cache = cache

    def __call__(
        self,
        time: Union[datetime.datetime, list[datetime.datetime]],
        channel: Union[str, list[str]],
    ) -> xr.DataArray:
        """Retrieve GFS initial data to be used for initial conditions for the given
        time, channel information, and optional history.

        Parameters
        ----------
        time : Union[datetime.datetime, list[datetime.datetime]]
            Timestamps to return data for.
        channel : str
            Channel(s) requested. Must be a subset of era5 available channels.

        Returns
        -------
        xr.DataArray
            GFS weather data array
        """
        if isinstance(channel, str):
            channel = [channel]

        if isinstance(time, datetime.datetime):
            time = [time]

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Fetch index file for requested time
        data_arrays = []
        for t0 in time:
            data_array = self.fetch_gfs_dataarray(t0, channel)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_gfs_dataarray(
        self,
        time: datetime.datetime,
        channels: list[str],
    ) -> xr.DataArray:
        """Retrives GFS data array for given date time by fetching the index file,
        fetching variable grib files and lastly combining grib files into single data
        array.

        Parameters
        ----------
        time : datetime.datetime
            Date time to fetch
        channels : list[str]
            List of atmosphric variables to fetch. Must be supported in GFS lexicon

        Returns
        -------
        xr.DataArray
            GFS data array for given date time

        Raises
        ------
        KeyError
            Un supported variable.
        """
        logger.debug(f"Fetching GFS index file: {time}")
        index_file = self._fetch_index(time)

        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        file_name = os.path.join(
            file_name, f"atmos/gfs.t{time.hour:0>2}z.pgrb2.0p25.f000"
        )
        grib_file_name = os.path.join(self.GFS_BUCKET_NAME, file_name)

        gfsda = xr.DataArray(
            data=np.empty((1, len(channels), len(self.GFS_LAT), len(self.GFS_LON))),
            dims=["time", "channel", "lat", "lon"],
            coords={
                "time": [time],
                "channel": channels,
                "lat": self.GFS_LAT,
                "lon": self.GFS_LON,
            },
        )

        for i, channel in enumerate(tqdm(channels, desc="Loading GFS channels")):
            # Convert from E2 MIP channel ID to GFS id and modifier
            try:
                gfs_name, modifier = GFSLexicon[channel]
            except KeyError:
                logger.warning(
                    f"Channel id {channel} not found in GFS lexicon, good luck"
                )
                gfs_name = channel

                def modifier(x: np.array) -> np.array:
                    return x

            if gfs_name not in index_file:
                raise KeyError(f"Could not find variable {gfs_name} in index file")

            byte_offset = index_file[gfs_name][0]
            byte_length = index_file[gfs_name][1]
            # Download the grib file to cache
            logger.debug(f"Fetching GFS grib file for channel: {channel} at {time}")
            grib_file = self._download_s3_grib_cached(
                grib_file_name, byte_offset=byte_offset, byte_length=byte_length
            )
            # Open into xarray data-array
            da = xr.open_dataarray(
                grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
            )
            gfsda[0, i] = modifier(da.values)

        return gfsda

    def _validate_time(self, times: list[datetime.datetime]) -> None:
        """Verify if date time is valid for GFS

        Parameters
        ----------
        times : list[datetime.datetime]
            List of date times to fetch data
        """
        for time in times:
            if not time.hour % 6 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 6 hour interval for GFS"
                )

            if time < datetime.datetime(year=2021, month=2, day=26):
                raise ValueError(
                    f"Requested date time {time} needs to be after Feburary 26th, 2021 for GFS"
                )

            if not self.available(time):
                raise ValueError(f"Requested date time {time} not available in GFS")

    def _fetch_index(self, time: datetime.datetime) -> dict[str, tuple[int, int]]:
        """Fetch GFS atmospheric index file

        Parameters
        ----------
        time : datetime.datetime
            Date time to fetch

        Returns
        -------
        dict[str, tuple[int, int]]
            Dictionary of GFS vairables (byte offset, byte length)
        """
        # https://www.nco.ncep.noaa.gov/pmb/products/gfs/
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}"
        file_name = os.path.join(
            file_name, f"atmos/gfs.t{time.hour:0>2}z.pgrb2.0p25.f000.idx"
        )
        s3_uri = os.path.join(self.GFS_BUCKET_NAME, file_name)
        # Grab index file
        index_file = self._download_s3_index_cached(s3_uri)
        with open(index_file, "r") as file:
            index_lines = [line.rstrip() for line in file]

        index_table = {}
        # Note we actually drop the last variable here (Vertical Speed Shear)
        for i, line in enumerate(index_lines[:-1]):
            lsplit = line.split(":")
            if len(lsplit) < 7:
                continue

            nlsplit = index_lines[i + 1].split(":")
            byte_length = int(nlsplit[1]) - int(lsplit[1])
            byte_offset = int(lsplit[1])
            key = f"{lsplit[3]}::{lsplit[4]}"
            if byte_length > self.MAX_BYTE_SIZE:
                raise ValueError(
                    f"Byte length, {byte_length}, of variable {key} larger than safe threshold of {self.MAX_BYTE_SIZE}"
                )

            index_table[key] = (byte_offset, byte_length)

        # Pop place holder
        return index_table

    def _download_s3_index_cached(self, path: str) -> str:
        sha = hashlib.sha256(path.encode())
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)
        fs = s3fs.S3FileSystem(anon=True)
        fs.get_file(path, cache_path)

        return cache_path

    def _download_s3_grib_cached(
        self, path: str, byte_offset: int = 0, byte_length: int = None
    ) -> str:
        sha = hashlib.sha256((path + str(byte_offset)).encode())
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)

        fs = s3fs.S3FileSystem(anon=True)
        if not pathlib.Path(cache_path).is_file():
            data = fs.read_block(path, offset=byte_offset, length=byte_length)
            with open(cache_path, "wb") as file:
                file.write(data)

        return cache_path

    @property
    def cache(self) -> str:
        cache_location = os.path.join(config.LOCAL_CACHE, "gfs")
        if not self._cache:
            cache_location = os.path.join(
                cache_location, f"tmp_{DistributedManager().rank}"
            )
        return cache_location

    @classmethod
    def available(
        cls,
        time: datetime.datetime,
    ) -> bool:
        """Checks if given date time is avaliable in the GFS object store

        Parameters
        ----------
        time : datetime.datetime
            Date time to access

        Returns
        -------
        bool
            If date time is avaiable
        """
        s3 = boto3.client(
            "s3", config=botocore.config.Config(signature_version=UNSIGNED)
        )
        # Object store directory for given time
        # Should contain two keys: atmos and wave
        file_name = f"gfs.{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}/"
        try:
            resp = s3.list_objects_v2(
                Bucket=cls.GFS_BUCKET_NAME, Prefix=file_name, Delimiter="/", MaxKeys=1
            )
        except botocore.exceptions.ClientError as e:
            logger.error("Failed to access from GFS S3 bucket")
            raise e

        return "KeyCount" in resp and resp["KeyCount"] > 0
