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
import io
import json
import os
import pathlib
import shutil
from typing import Any, List, Optional, Union

import fsspec
import numpy as np
import s3fs
import xarray as xr
from loguru import logger
from modulus.distributed.manager import DistributedManager
from modulus.utils.filesystem import _download_cached

from earth2mip import config
from earth2mip.networks.fcnv2_sm import CHANNELS


class ERA5H5:
    """HDF5 ERA5 reanalysis data source provided on a equirectangular grid. This is a
    data source that is largely used by Nvidia for training large scale global weather
    models as well as scoring.

    Works with a directory structure like this:

        data.json
        subdirA/2018.h5
        subdirB/2017.h5
        subdirB/2016.h5
        2015.h5

    The H5 files should have the followings data:

        fields: [time, channel, lat, lon]

    data.json should have fields

        coords.channel - list of channels (default 73 fcn v2 channel set)
        coords.lat - list of latitude coords (default 0.25 degree)
        coords.lon - list of longitude coords (default 0.25 degree)
        dt_hours - timestep in hours (default 6 hours)

    Parameters
    ----------
    file_path : str
        Directory to folder containing HDF5 files (will search two directories deep).
        Presently supports local file systems and s3 object storage
    meta_data : Optional[dict], optional
        HDF5 meta data containing coordinate information. Will override defaults and any
        values in data.json, by default None
    cache : bool, optional
        Cache data if remote, by default True

    Warning
    -------
    When using a s3 object store. This is a remote data source and can potentially
    download a large amount of data to your local machine for large requests.
    """

    dt_hours: float = 6.0
    channel: list[str] = CHANNELS
    ERA5_LAT: np.array = np.linspace(90, -90, 721)
    ERA5_LON: np.array = np.linspace(0, 359.75, 1440)

    def __init__(
        self, file_path: str, meta_data: Optional[dict] = None, cache: bool = True
    ):
        self.file_path = file_path
        self._cache = cache
        self._load_meta_data()

    def __call__(
        self,
        time: Union[datetime.datetime, List[datetime.datetime]],
        channel: Union[str, List[str]],
    ) -> xr.DataArray:
        """Retrieve data from ERA5 (either mounted on NGC or PBSS) from a particular
        date and including n_history periods into the past from that date.

        Parameters
        ----------
        time : Union[datetime.datetime, list[datetime.datetime]]
            Timestamps to return data for.
        channel : str
            Channel(s) requested. Must be a subset of era5 available channels.

        Returns
        -------
        xr.DataArray
            ERA5 weather data array from H5 files
        """

        if isinstance(channel, str):
            channel = [channel]

        if isinstance(time, datetime.datetime):
            time = [time]

        # Create cache dir if doesnt exist
        pathlib.Path(self.cache).mkdir(parents=True, exist_ok=True)

        fs = self._get_fs()
        # Search for H5 files in provided directory
        h5_files = fs.glob(os.path.join(self.file_path, "**.h5"), maxdepth=2)
        files = {int(pathlib.Path(f).stem): f for f in h5_files}
        logger.debug(f"Discoverd {len(files)} H5 files")

        self._validate_time(time, files)

        # Open H5 files
        da = xr.concat(
            [self._open_era5_hdf5(fs.open(files[t.year]), t, channel) for t in time],
            "time",
        )

        # Delete cache if needed
        if not self._cache:
            shutil.rmtree(self.cache)

        return da

    def _load_meta_data(self, meta_data: Optional[dict] = None) -> None:

        if meta_data is None:
            metadata_path = os.path.join(self.file_path, "data.json")
            metadata_path = _download_cached(metadata_path, local_cache_path=self.cache)
            with open(metadata_path) as mf:
                meta_data = json.load(mf)

        # Update default values if provided
        if "dt_hours" in meta_data:
            self.dt_hours = float(meta_data["dt_hours"])

        if "coords" in meta_data:
            if "channel" in meta_data["coords"]:
                self.channel = meta_data["coords"]["channel"]
            if "lat" in meta_data["coords"]:
                self.ERA5_LAT = np.array(meta_data["coords"]["lat"])
            if "lon" in meta_data["coords"]:
                self.ERA5_LON = np.array(meta_data["coords"]["lon"])

    @property
    def cache(self) -> str:
        cache_location = os.path.join(config.LOCAL_CACHE, "era5h5")
        if not self._cache:
            cache_location = os.path.join(
                cache_location, f"tmp_{DistributedManager().rank}"
            )
        return cache_location

    def _validate_time(
        self, times: list[datetime.datetime], files: dict[int, Any]
    ) -> None:
        """Pre-checks to verify the requested times are valid for ERA5 H5 files

        Parameters
        ----------
        times : list[datetime.datetime]
            List of date times to fetch data
        files : dict[int, Any]
            Dictionary of discovered H5 files
        """
        for time in times:
            if not time.hour % self.dt_hours == 0:
                raise ValueError(
                    f"Requested date time {time} needs to be {self.dt_hours} hour interval"
                )

            if time.year not in files:
                raise ValueError(
                    f"Requested date year {time.year} has no corresponding H5 file"
                )

    # Not using modulus here because of cache location spec
    def _get_fs(self) -> fsspec.spec.AbstractFileSystem:
        if self.file_path.startswith("s3://"):
            return s3fs.S3FileSystem(
                client_kwargs=dict(endpoint_url=config.S3_ENDPOINT)
            )
        else:
            return fsspec.filesystem("file")

    def _open_era5_hdf5(
        self,
        file: io.IOBase,
        date: datetime.datetime,
        channel: Union[str, List[str]],
    ) -> xr.DataArray:
        """Helper function to open ERA5 data stored in h5netcdf file."""

        logger.debug(f"Loading data array for time {date}")
        time_step = datetime.timedelta(hours=self.dt_hours)
        da = xr.open_dataarray(
            file, engine="h5netcdf", phony_dims="sort", cache=self._cache
        )
        da = da.rename(
            {
                da.dims[0]: "time",
                da.dims[1]: "channel",
                da.dims[2]: "lat",
                da.dims[3]: "lon",
            }
        )
        da = da.assign_coords(
            time=[
                datetime.datetime(date.year, 1, 1, 0, 0) + time_step * int(i)
                for i in da.time
            ],
            channel=self.channel,
            lat=self.ERA5_LAT,
            lon=self.ERA5_LON,
        )
        return da.sel(time=[date], channel=channel)


if __name__ == "__main__":

    ds = ERA5H5("/mount/73vars/")

    channel = ["t2m"]
    time = datetime.datetime(year=2017, month=12, day=4, hour=6)

    da = ds(time, channel)
    print(da)
