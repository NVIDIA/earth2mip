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
import hashlib
import os
import pathlib
import shutil
from typing import Union

import boto3
import botocore
import ecmwf.opendata
import numpy as np
import xarray as xr
from botocore import UNSIGNED
from loguru import logger
from modulus.distributed.manager import DistributedManager
from tqdm import tqdm

from earth2mip import config
from earth2mip.lexicon import IFSLexicon

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class IFS:
    """The integrated forecast system (IFS) initial state data source provided on an
    equirectangular grid. This data is part of ECMWF's open data project on AWS. This
    data source is provided on a 0.4 degree lat lon grid at 6-hour intervals spanning
    from Jan 18th 2023 to present date.

    Parameters
    ----------
    cache : bool, optional
        Cache data source on local memory, by default True
    verbose : bool, optional
        Print download progress, by default True

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    This data source only fetches the initial state of control forecast of IFS and does
    not fetch an predicted time steps.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://confluence.ecmwf.int/display/DAC/ECMWF+open+data%3A+real-time+forecasts
    - https://registry.opendata.aws/ecmwf-forecasts/
    """

    IFS_BUCKET_NAME = "ecmwf-forecasts"
    IFS_LAT = np.linspace(90, -90, 451)
    IFS_LON = np.linspace(0, 359.6, 900)

    def __init__(self, cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._verbose = verbose
        self.client = ecmwf.opendata.Client(source="aws")

    def __call__(
        self,
        time: Union[datetime.datetime, list[datetime.datetime]],
        channel: Union[str, list[str]],
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        t : datetime.datetime or list[datetime.datetime]
            Timestamps to return data for (UTC).
        channel : str or list[str]
            Strings or list of strings that refer to the
            channel/variables to return.

        Returns
        -------
        xr.DataArray
            IFS weather data array
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
            data_array = self.fetch_ifs_dataarray(t0, channel)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_ifs_dataarray(
        self,
        time: datetime.datetime,
        channels: list[str],
    ) -> xr.DataArray:
        """Retrives IFS data array for given date time by fetching variable grib files
        using the ecmwf opendata package and combining grib files into a data array.

        Parameters
        ----------
        time : datetime.datetime
            Date time to fetch
        channels : list[str]
            List of atmosphric variables to fetch. Must be supported in IFS lexicon

        Returns
        -------
        xr.DataArray
            IFS data array for given date time
        """
        ifsda = xr.DataArray(
            data=np.empty((1, len(channels), len(self.IFS_LAT), len(self.IFS_LON))),
            dims=["time", "channel", "lat", "lon"],
            coords={
                "time": [time],
                "channel": channels,
                "lat": self.IFS_LAT,
                "lon": self.IFS_LON,
            },
        )

        # TODO: Add MP here, can further optimize by combining pressure levels
        # Not doing until tested.
        for i, channel in enumerate(
            tqdm(channels, desc="Loading IFS channels", disable=(not self._verbose))
        ):
            # Convert from E2 MIP channel ID to GFS id and modifier
            try:
                ifs_name, modifier = IFSLexicon[channel]
            except KeyError as e:
                logger.error(f"Channel id {channel} not found in IFS lexicon")
                raise e

            variable, levtype, level = ifs_name.split("::")

            logger.debug(f"Fetching IFS grib file for channel: {channel} at {time}")
            grib_file = self._download_ifs_grib_cached(variable, levtype, level, time)
            # Open into xarray data-array
            # Provided [-180, 180], roll to [0, 360]
            da = xr.open_dataarray(
                grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
            ).roll(longitude=-len(self.IFS_LON) // 2)
            ifsda[0, i] = modifier(da.values)

        return ifsda

    def _validate_time(self, times: list[datetime.datetime]) -> None:
        """Verify if date time is valid for IFS

        Parameters
        ----------
        times : list[datetime.datetime]
            List of date times to fetch data
        """
        for time in times:
            if not time.hour % 6 == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 6 hour interval for IFS"
                )

            if time < datetime.datetime(year=2023, month=1, day=18):
                raise ValueError(
                    f"Requested date time {time} needs to be after January 18th, 2023 for IFS"
                )

            if not self.available(time):
                raise ValueError(f"Requested date time {time} not available in IFS")

    def _download_ifs_grib_cached(
        self,
        variable: str,
        levtype: str,
        level: Union[str, list[str]],
        time: datetime.datetime,
    ) -> str:
        if isinstance(level, str):
            level = [level]

        sha = hashlib.sha256(f"{variable}_{levtype}_{'_'.join(level)}_{time}".encode())
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            request = {
                "type": "fc",
                "param": variable,
                "levtype": levtype,
                "step": 0,  # Would change this for forecasts
                "target": cache_path,
            }
            if levtype == "pl":  # Pressure levels
                request["levelist"] = level
            # Download
            self.client.retrieve(**request)

        return cache_path

    @property
    def cache(self) -> str:
        cache_location = os.path.join(config.LOCAL_CACHE, "ifs")
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
        """Checks if given date time is avaliable in the IFS AWS data store

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
        file_name = f"{time.year}{time.month:0>2}{time.day:0>2}/{time.hour:0>2}z/"
        try:
            resp = s3.list_objects_v2(
                Bucket=cls.IFS_BUCKET_NAME, Prefix=file_name, Delimiter="/", MaxKeys=1
            )
        except botocore.exceptions.ClientError as e:
            logger.error("Failed to access from IFS S3 bucket")
            raise e

        return "KeyCount" in resp and resp["KeyCount"] > 0
