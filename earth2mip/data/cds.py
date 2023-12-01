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
from time import sleep
from typing import Union

import cdsapi
import numpy as np
import xarray as xr
from loguru import logger
from modulus.distributed.manager import DistributedManager
from tqdm import tqdm

from earth2mip import config
from earth2mip.lexicon import CDSLexicon

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class CDS:
    """The climate data source (CDS) serving ERA5 re-analysis data.

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
    https://cds.climate.copernicus.eu/cdsapp#!/home
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
            ERA5 weather data array from CDS
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
            data_array = self.fetch_cds_dataarray(t0, channel)
            data_arrays.append(data_array)

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return xr.concat(data_arrays, dim="time")

    def fetch_cds_dataarray(
        self,
        time: datetime.datetime,
        channels: list[str],
    ) -> xr.DataArray:
        """Retrives CDS data array for given date time by fetching variable grib files
        using the cdsapi package and combining grib files into a single data array.

        Parameters
        ----------
        time : datetime.datetime
            Date time to fetch
        channels : list[str]
            List of atmosphric variables to fetch. Must be supported in CDS lexicon

        Returns
        -------
        xr.DataArray
            CDS data array for given date time
        """
        cdsda = xr.DataArray(
            data=np.empty((1, len(channels), len(self.CDS_LAT), len(self.CDS_LON))),
            dims=["time", "channel", "lat", "lon"],
            coords={
                "time": [time],
                "channel": channels,
                "lat": self.CDS_LAT,
                "lon": self.CDS_LON,
            },
        )

        # TODO: Add MP here, can further optimize by combining pressure levels
        # Not doing until tested.
        for i, channel in enumerate(
            tqdm(channels, desc="Loading CDS channels", disable=(not self._verbose))
        ):
            # Convert from E2 MIP channel ID to GFS id and modifier
            try:
                cds_name, modifier = CDSLexicon[channel]  # type: ignore[misc]
            except KeyError as e:
                logger.error(
                    f"Channel id {channel} not found in CDS lexicon, good luck"
                )
                raise e

            dataset_name, variable, level = cds_name.split("::")

            logger.debug(f"Fetching CDS grib file for channel: {channel} at {time}")
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


if __name__ == "__main__":

    ds = CDS()

    time = datetime.datetime(year=2022, month=2, day=1)
    channel = [
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
    ]

    da = ds(time, channel)

    # Create a sample NumPy array with dimensions [73, 721, 1440]
    # Replace this with your actual NumPy array
    data_array = da.values

    # Determine the number of rows and columns for the subplots
    num_rows = 10  # You can adjust this based on your preference
    num_cols = 8  # You can adjust this based on your preference

    import matplotlib.pyplot as plt

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 20))

    # Plot contours for each subplot
    for i in range(min(num_rows * num_cols, data_array.shape[1])):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]

        contour = ax.contourf(data_array[0, i, :, :], cmap="viridis", vmin=0, vmax=105)
        ax.set_title(f'Contour {da.coords["channel"][i].values}')

        # Add colorbar for the last column of subplots
        cbar = plt.colorbar(contour, ax=ax, orientation="vertical", shrink=0.8)
        cbar.set_label("Values")

    # Adjust layout and show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig("300dpi.png", dpi=500)

    print(da)

# def _get_cds_requests(codes, time, format):
#     grid = (0.25, 0.25)
#     area = (90, -180, -90, 180)

#     # create a list of arguments for each call to retrieve_channel_data
#     levels = set()
#     pressure_level_names = set()
#     single_level_names = set()
#     for v in codes:
#         if isinstance(v, PressureLevelCode):  # it's a pressure level variable
#             levels.add(v.level)
#             pressure_level_names.add(v.id)
#         elif isinstance(v, SingleLevelCode):  # it's a single level variable
#             single_level_names.add(v.id)

#     if pressure_level_names and levels:
#         # TODO to limit download size for many levels, split this into one
#         # request per variable if there are more than some number of levels.
#         yield (
#             "reanalysis-era5-pressure-levels",
#             {
#                 "product_type": "reanalysis",
#                 "variable": list(pressure_level_names),
#                 "pressure_level": sorted(levels),
#                 "year": time.strftime("%Y"),
#                 "month": time.strftime("%m"),
#                 "day": time.strftime("%d"),
#                 "time": time.strftime("%H:%M"),
#                 "area": area,
#                 "grid": grid,
#                 "format": format,
#             },
#         )

#     if single_level_names:
#         yield (
#             "reanalysis-era5-single-levels",
#             {
#                 "product_type": "reanalysis",
#                 "variable": sorted(single_level_names),
#                 "year": time.strftime("%Y"),
#                 "month": time.strftime("%m"),
#                 "day": time.strftime("%d"),
#                 "time": time.strftime("%H:%M"),
#                 "area": area,
#                 "grid": grid,
#                 "format": format,
#             },
#         )

#     def _parse_files(
#         self, codes: List[Union[SingleLevelCode, PressureLevelCode]], files: List[str]
#     ) -> xarray.DataArray:
#         """Retrieve ``codes`` from a list of ``files``

#         Returns:
#             a data array of all the codes

#         """
#         arrays = [None] * len(codes)
#         for path in files:
#             with open(path) as f:
#                 while True:
#                     gid = eccodes.codes_grib_new_from_file(f)
#                     if gid is None:
#                         break
#                     id = eccodes.codes_get(gid, "paramId")
#                     level = eccodes.codes_get(gid, "level")
#                     type_of_level = eccodes.codes_get(gid, "typeOfLevel")

#                     if type_of_level == "surface":
#                         code = SingleLevelCode(id)
#                     else:
#                         code = PressureLevelCode(id, level=level)

#                     nlat = eccodes.codes_get(gid, "Nj")
#                     nlon = eccodes.codes_get(gid, "Ni")

#                     lat = eccodes.codes_get_array(gid, "latitudes").reshape(nlat, nlon)
#                     lon = eccodes.codes_get_array(gid, "longitudes").reshape(nlat, nlon)
#                     vals = eccodes.codes_get_values(gid).reshape(nlat, nlon)
#                     eccodes.codes_release(gid)

#                     try:
#                         i = codes.index(code)
#                     except ValueError:
#                         continue

#                     arrays[i] = vals
#         array = np.stack(arrays)
#         coords = {}
#         coords["lat"] = lat[:, 0]
#         coords["lon"] = lon[0, :]
#         return xarray.DataArray(array, dims=["channel", "lat", "lon"], coords=coords)

#     def _download_codes(self, client, codes, time, d) -> xarray.DataArray:
#         files = []
#         format = "grib"

#         def download(arg):
#             name, req = arg
#             hash_ = hashlib.sha256(str(req).encode()).hexdigest()
#             dirname = os.path.join(d, hash_)
#             os.makedirs(dirname, exist_ok=True)
#             filename = name + ".grib"
#             path = os.path.join(dirname, filename)
#             if not os.path.exists(path):
#                 logger.info(f"Data not found in cache. Downloading {name} to {path}")
#                 client.retrieve(name, req, path + ".tmp")
#                 shutil.move(path + ".tmp", path)
#             else:
#                 logger.info(f"Found data in cache. Using {path}.")
#             return path

#         requests = _get_cds_requests(codes, time, format)
#         with ThreadPoolExecutor(4) as pool:
#             files = pool.map(download, requests)

#         return _parse_files(codes, files)

#     def _get_channels(self, client, time: datetime.datetime, channels: List[str], d):
#         codes = [parse_channel(c) for c in channels]
#         darray = _download_codes(client, codes, time, d)
#         return (
#             darray.assign_coords(channel=channels)
#             .assign_coords(time=time)
#             .transpose("channel", "lat", "lon")
#             .assign_coords(lon=darray["lon"] + 180.0)
#             .roll(lon=1440 // 2)
#         )
