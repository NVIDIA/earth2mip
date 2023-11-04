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

import contextlib
import datetime
import glob
import json
import os
import pathlib
import tempfile
from typing import Any, Iterable, Optional

import h5py
import numpy as np
import xarray

from earth2mip.datasets.era5 import time

__all__ = ["open_34_vars", "open_hdf5"]

METADATA = pathlib.Path(__file__).parent / "data.json"


def open_hdf5(*, path, f=None, metadata):
    dims = metadata["dims"]
    h5_path = metadata["h5_path"]
    time_step_hours = metadata.get("dhours", 6)
    time_step = datetime.timedelta(hours=time_step_hours)

    ds = xarray.open_dataset(f or path, engine="h5netcdf", phony_dims="sort")
    array = ds[h5_path]
    ds = array.rename(dict(zip(array.dims, dims)))
    year = time.filename_to_year(path)
    n = array.shape[0]
    ds = ds.assign_coords(
        time=time.datetime_range(year, time_step=time_step, n=n), **metadata["coords"]
    )
    ds = ds.assign_attrs(metadata["attrs"], path=path)
    return ds


@contextlib.contextmanager
def open_all_hdf5(root: str) -> Iterable[xarray.DataArray]:
    """A context manager to open hdf5 ERA5 data as a single logical xarray

    Args:
        root: A **local** directory where the dataset is stored. Metadata should
            be stored at ``root/data.json``. HDF5 data will be read from
            subdirectories, typically ``train``, ``test``, and
            ``out_of_sample``.

    Returns:
        an xarray dataset

    """

    try:
        metadata_path = pathlib.Path(root) / "data.json"
        metadata = json.loads(metadata_path.read_text())
    except FileNotFoundError:
        metadata = json.loads(METADATA.read_text())

    with tempfile.NamedTemporaryFile("wb") as f:
        _create_virtual_dataset(root, f.name)

        with xarray.open_dataset(f.name, chunks=None) as ds:
            dims = ["year", "step"] + metadata["dims"][1:]
            ds = ds.rename(dict(zip(ds.dims, dims)))

            step = np.timedelta64(6, "h") * np.arange(ds.sizes["step"])
            ds = ds.assign_coords(step=step).assign_coords(metadata["coords"])
            yield ds.fields


def _create_virtual_dataset(root: str, virtual_dataset_path: str):
    file_paths = glob.glob(root + "/*/*.h5")
    file_paths = sorted(file_paths, key=os.path.basename)

    # Open the first file to extract the dataset shape
    with h5py.File(file_paths[0], "r") as f:
        dataset_shape = f["fields"].shape

    # Create the virtual dataset
    with h5py.File(virtual_dataset_path, "w", libver="latest") as f:
        # Define the virtual dataset layout
        layout = h5py.VirtualLayout(shape=(len(file_paths),) + dataset_shape, dtype="f")
        year_d = f.create_dataset("year", shape=len(file_paths), dtype="i")
        for i, file_path in enumerate(file_paths):
            # Define the virtual source dataset
            source = h5py.VirtualSource(file_path, "fields", shape=dataset_shape)
            # Assign the virtual source dataset to the virtual layout
            layout[i, ...] = source
            filename = os.path.basename(file_path)
            base, _ = os.path.splitext(filename)
            year_d[i] = int(base)

        # Create the virtual dataset
        f.create_virtual_dataset("fields", layout)


def open_34_vars(path: str, f: Optional[Any] = None) -> xarray.DataArray:
    """Open 34Vars hdf5 file

    Args:
        path: local path to hdf5 file
        f: an optional file-like object to load the data from. Useful for
            remote data and fsspec.

    Examples:

        >>> import earth2mip.datasets
        >>> path = "/out_of_sample/2018.h5"
        >>> datasets.era5.open_34_vars(path)
        <xarray.DataArray 'fields' (time: 1460, channel: 34, lat: 721, lon: 1440)>
        dask.array<array, shape=(1460, 34, 721, 1440), dtype=float32, chunksize=(1, 1, 721, 1440), chunktype=numpy.ndarray> # noqa
        Coordinates:
        * time     (time) datetime64[ns] 2018-01-01 ... 2018-12-31T18:00:00
        * lat      (lat) float64 90.0 89.75 89.5 89.25 ... -89.25 -89.5 -89.75 -90.0
        * lon      (lon) float64 0.0 0.25 0.5 0.75 1.0 ... 359.0 359.2 359.5 359.8
        * channel  (channel) <U5 'u10' 'v10' 't2m' 'sp' ... 'v900' 'z900' 't900'
        Attributes:
            selene_path:  /lustre/fsw/sw_climate_fno/34Var
            description:  ERA5 data at 6 hourly frequency with snapshots at 0000, 060...
            path:         /out_of_sample/2018.h5
    """

    metadata = json.loads(METADATA.read_text())
    return open_hdf5(path=path, f=f, metadata=metadata)
