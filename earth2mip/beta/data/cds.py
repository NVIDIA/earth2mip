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
from time import sleep
from typing import Union

import cdsapi
import numpy as np
import xarray as xr
from loguru import logger
from modulus.distributed.manager import DistributedManager
from tqdm import tqdm

from earth2mip import config
from earth2mip.beta.lexicon import CDSLexicon

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class CDS:
    """The climate data source (CDS) serving ERA5 re-analysis data. This data soure
    requires users to have a CDS API access key which can be obtained for free on the
    CDS webpage.

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
    Additional information on the data repository can be referenced here:

    - https://cds.climate.copernicus.eu/cdsapp#!/home
    """

    MAX_BYTE_SIZE = 20000000

    CDS_LAT = np.linspace(90, -90, 721)
    CDS_LON = np.linspace(0, 359.75, 1440)

    def __init__(self, cache: bool = True, verbose: bool = True):
        self._cache = cache
        self._verbose = verbose
        self.cds_client = cdsapi.Client(
            debug=False, quiet=True, wait_until_complete=False
        )

    def __call__(
        self,
        time: Union[datetime.datetime, list[datetime.datetime]],
        variable: Union[str, list[str]],
    ) -> xr.DataArray:
        """Function to get data.

        Parameters
        ----------
        time : Union[datetime.datetime, List[datetime.datetime]]
            Timestamps to return data for (UTC).
        variable : Union[str, List[str]]
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from CDS
        """
        if isinstance(variable, str):
            variable = [variable]

        if isinstance(time, datetime.datetime):
            time = [time]

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        # Make sure input time is valid
        self._validate_time(time)

        # Fetch index file for requested time
        data_arrays = []
        for t0 in time:
            data_array = self.fetch_cds_dataarray(t0, variable)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_cds_dataarray(
        self,
        time: datetime.datetime,
        variables: list[str],
    ) -> xr.DataArray:
        """Retrives CDS data array for given date time by fetching variable grib files
        using the cdsapi package and combining grib files into a single data array.

        Parameters
        ----------
        time : datetime.datetime
            Date time to fetch
        variables : list[str]
            List of atmosphric variables to fetch. Must be supported in CDS lexicon

        Returns
        -------
        xr.DataArray
            CDS data array for given date time
        """
        cdsda = xr.DataArray(
            data=np.empty((1, len(variables), len(self.CDS_LAT), len(self.CDS_LON))),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": [time],
                "variable": variables,
                "lat": self.CDS_LAT,
                "lon": self.CDS_LON,
            },
        )

        # TODO: Add MP here, can further optimize by combining pressure levels
        # Not doing until tested.
        for i, variable in enumerate(
            tqdm(variables, desc="Loading CDS variables", disable=(not self._verbose))
        ):
            # Convert from E2 MIP variable ID to GFS id and modifier
            try:
                cds_name, modifier = CDSLexicon[variable]
            except KeyError as e:
                logger.error(f"variable id {variable} not found in CDS lexicon")
                raise e

            dataset_name, variable, level = cds_name.split("::")

            logger.debug(f"Fetching CDS grib file for variable: {variable} at {time}")
            grib_file = self._download_cds_grib_cached(
                dataset_name, variable, level, time
            )
            # Open into xarray data-array
            da = xr.open_dataarray(
                grib_file, engine="cfgrib", backend_kwargs={"indexpath": ""}
            )
            cdsda[0, i] = modifier(da.values)

        return cdsda

    def _validate_time(self, times: list[datetime.datetime]) -> None:
        """Verify if date time is valid for CDS

        Parameters
        ----------
        times : list[datetime.datetime]
            List of date times to fetch data
        """
        for time in times:
            if not time.minute == 0 and time.second == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be 1 hour interval for CDS"
                )

            if time < datetime.datetime(year=1940, month=1, day=1):
                raise ValueError(
                    f"Requested date time {time} needs to be after January 1st, 1940 for CDS"
                )

            if not self.available(time):
                raise ValueError(f"Requested date time {time} not available in CDS")

    def _download_cds_grib_cached(
        self,
        dataset_name: str,
        variable: str,
        level: Union[str, list[str]],
        time: datetime.datetime,
    ) -> str:
        if isinstance(level, str):
            level = [level]

        sha = hashlib.sha256(
            f"{dataset_name}_{variable}_{'_'.join(level)}_{time}".encode()
        )
        filename = sha.hexdigest()

        cache_path = os.path.join(self.cache, filename)

        if not pathlib.Path(cache_path).is_file():
            # Assemble request
            rbody = {
                "variable": variable,
                "product_type": "reanalysis",
                # "date": "2017-12-01/2017-12-02", (could do time range)
                "year": time.year,
                "month": time.month,
                "day": time.day,
                "time": time.strftime("%H:00"),
                "format": "grib",
            }
            if dataset_name == "reanalysis-era5-pressure-levels":
                rbody["pressure_level"] = level
            r = self.cds_client.retrieve(dataset_name, rbody)
            # Queue request
            while True:
                r.update()
                reply = r.reply
                logger.debug(
                    f"Request ID:{reply['request_id']}, state: {reply['state']}"
                )
                if reply["state"] == "completed":
                    break
                elif reply["state"] in ("queued", "running"):
                    logger.debug(f"Request ID: {reply['request_id']}, sleeping")
                    sleep(0.5)
                elif reply["state"] in ("failed",):
                    logger.error(
                        f"CDS request fail for: {dataset_name} {variable} {level} {time}"
                    )
                    logger.error(f"Message: {reply['error'].get('message')}")
                    logger.error(f"Reason: {reply['error'].get('reason')}")
                    raise Exception("%s." % (reply["error"].get("message")))
            # Download when ready
            r.download(cache_path)

        return cache_path

    @property
    def cache(self) -> str:
        cache_location = os.path.join(config.LOCAL_CACHE, "cds")
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
        """Checks if given date time is avaliable in the CDS with the pressure level
        database

        Parameters
        ----------
        time : datetime.datetime
            Date time to access

        Returns
        -------
        bool
            If date time is avaiable
        """
        client = cdsapi.Client(debug=False, quiet=True, wait_until_complete=False)
        # Assemble request
        r = client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "variable": "u_component_of_wind",
                "product_type": "reanalysis",
                "pressure_level": ["50"],
                "year": time.year,
                "month": time.month,
                "day": time.day,
                "time": time.strftime("%H:00"),
                "format": "grib",
            },
        )
        # Queue request
        while True:
            r.update()
            reply = r.reply
            logger.debug(f"Request ID:{reply['request_id']}, state: {reply['state']}")
            if reply["state"] == "completed":
                break
            elif reply["state"] in ("queued", "running"):
                logger.debug(f"Request ID: {reply['request_id']}, sleeping")
                sleep(0.5)
            elif reply["state"] in ("failed",):
                logger.error(f"CDS request fail for {time}")
                logger.error(f"Message: {reply['error'].get('message')}")
                logger.error(f"Reason: {reply['error'].get('reason')}")
                return False

        return True
