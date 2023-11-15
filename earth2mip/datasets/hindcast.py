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

"""Routines for opening fcn-mip hindcast outputs
"""
import datetime
import json
import os

import pandas as pd
import xarray
from zarr.storage import FSStore

from earth2mip import filesystem
from earth2mip.datasets.zarr_directory import NestedDirectoryStore


def open_forecast(root, group, chunks=None):
    """Open a fcn-mip forecast as single xarray object

    The directory structure should contain items like this:

        {root}/2018-01-01T00:00:00/mean.zarr/
        {root}/2018-01-02T00:00:00/mean.zarr/

    """
    if isinstance(root, str):
        map_ = FSStore(url=root)
    else:
        map_ = root

    config_path = os.path.join(root, "config.json")
    local_config = filesystem.download_cached(config_path)
    with open(local_config) as f:
        config = json.load(f)
    items = config["protocol"]["times"]

    times = []
    for f in items:
        try:
            datetime.datetime.fromisoformat(f)
        except ValueError:  # noqa
            pass
        else:
            times.append(f)
    times = sorted(times)

    store = NestedDirectoryStore(
        map=map_,
        group=group,
        directories=items,
        concat_dim="initial_time",
        static_coords=("lat", "lon"),
        dim_rename={"time": "lead_time"},
    )

    # TODO this only works locally
    example = xarray.open_zarr(os.path.join(root, f"{items[0]}/{group}"))
    ds = xarray.open_zarr(store, chunks=None).assign_coords(
        {dim: example[dim] for dim in store.static_coords}
    )
    ds["initial_time"] = pd.to_datetime(ds.initial_time)

    if "time" in example.variables:
        ds["lead_time"] = (ds.time - ds.initial_time).isel(initial_time=0)
        ds = ds.rename(time="valid_time", lead_time="time")
    return ds
