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
import json
import logging
import os
import warnings
from typing import Any, List, Optional

import numpy as np
import s3fs
import xarray

from earth2mip import config, filesystem, grid
from earth2mip.datasets import era5
from earth2mip.initial_conditions import base

__all__ = ["open_xarray", "DataSource"]

logger = logging.getLogger(__name__)
# TODO move to earth2mip/datasets/era5?


class DataSource(base.DataSource):
    """HDF5 Data Sources

    Works with a directory structure like this::

        data.json
        subdirA/2018.h5
        subdirB/2017.h5
        subdirB/2016.h5

    data.json should have fields

        h5_path - the name of the data within the hdf5 file
        coords.channel - list of channels
        coords.lat - list of lats
        coords.lon - list of lons
        dhours - timestep in hours (default 6 hours)

    """

    def __init__(
        self, root: str, metadata: Any, channel_names: Optional[List[str]] = None
    ):
        """

        Args:
            root: Path to the root of the HDF5 data.
            metadata: Metadata about the HDF5 data.
            channel_names: If provided, only get these channel names.
                Defaults to all channels in the data.
        """
        self.root = root
        self.metadata = metadata
        if channel_names is None:
            self._channel_names = metadata["coords"]["channel"]
        else:
            self._channel_names = [
                c for c in metadata["coords"]["channel"] if c in channel_names
            ]

    @classmethod
    def from_path(cls, root: str, **kwargs: Any) -> "DataSource":
        metadata_path = os.path.join(root, "data.json")
        metadata_path = filesystem.download_cached(metadata_path)
        with open(metadata_path) as mf:
            metadata = json.load(mf)
        return cls(root, metadata, **kwargs)

    @property
    def grid(self):
        return grid.LatLonGrid(
            lat=self.metadata["coords"]["lat"], lon=self.metadata["coords"]["lon"]
        )

    @property
    def channel_names(self):
        return self._channel_names

    @property
    def time_means(self):
        time_mean_path = os.path.join(self.root, "stats", "time_means.npy")
        time_mean_path = filesystem.download_cached(time_mean_path)
        return np.load(time_mean_path)

    def __getitem__(self, time: datetime.datetime) -> np.ndarray:
        path = _get_path(self.root, time)
        if path.startswith("s3://"):
            fs = s3fs.S3FileSystem(
                client_kwargs=dict(endpoint_url="https://pbss.s8k.io")
            )
            f = fs.open(path)
        else:
            f = None

        logger.debug(f"Opening {path} for {time}.")
        ds = era5.open_hdf5(path=path, f=f, metadata=self.metadata)
        subset = ds.sel(time=time, channel=self._channel_names)
        return subset.values


def _get_path(path: str, time) -> str:
    filename = time.strftime("%Y.h5")
    h5_files = filesystem.glob(os.path.join(path, "**/*.h5"), maxdepth=2)
    files = {os.path.basename(f): f for f in h5_files}
    return files[filename]


def open_xarray(time: datetime.datetime) -> xarray.DataArray:
    warnings.warn(DeprecationWarning("This function will be removed"))
    root = config.ERA5_HDF5
    path = _get_path(root, time)
    logger.debug(f"Opening {path} for {time}.")

    if path.endswith(".h5"):
        if path.startswith("s3://"):
            fs = s3fs.S3FileSystem(
                client_kwargs=dict(endpoint_url="https://pbss.s8k.io")
            )
            f = fs.open(path)
        else:
            f = None

        metadata_path = os.path.join(config.ERA5_HDF5, "data.json")
        metadata_path = filesystem.download_cached(metadata_path)
        with open(metadata_path) as mf:
            metadata = json.load(mf)
        ds = era5.open_hdf5(path=path, f=f, metadata=metadata)

    elif path.endswith(".nc"):
        ds = xarray.open_dataset(path).fields

    return ds
