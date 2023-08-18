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

import os
import datetime
import xarray
import json
from earth2mip import filesystem, schema, config
import logging
import numpy as np
import h5py

logger = logging.getLogger(__name__)


def _get_path(path: str, time) -> str:
    filename = time.strftime("%Y.h5")
    h5_files = filesystem.glob(os.path.join(path, "*.h5"))
    files = {os.path.basename(f): f for f in h5_files}
    return files[filename]


def _get_time(time: datetime.datetime) -> int:
    day_of_year = time.timetuple().tm_yday - 1
    hour_of_day = time.timetuple().tm_hour
    hours_since_jan_01 = 24 * day_of_year + hour_of_day
    return int(hours_since_jan_01 / 6)


def _get_hdf5(path: str, metadata, time: datetime.datetime) -> xarray.DataArray:
    dims = metadata["dims"]
    h5_path = metadata["h5_path"]
    variables = []
    ic = _get_time(time)
    with h5py.File(path, "r") as f:
        for nm in h5_path:
            if nm == "pl":
                pl = f[nm][ic : ic + 1]
            elif nm == "sl":
                sl = f[nm][ic : ic + 1]

    assert "pl" in locals() and "sl" in locals()

    pl_list = []
    for var_idx in range(pl.shape[1]):
        pl_list.append(pl[:, var_idx])
    pl = np.concatenate(pl_list, axis=1)  # pressure level vars flattened
    data = np.concatenate([pl, sl], axis=1)
    ds = xarray.DataArray(
        data,
        dims=["time", "channel", "lat", "lon"],
        coords={
            "time": [time],
            "channel": metadata["coords"]["channel"],
            "lat": metadata["coords"]["lat"],
            "lon": metadata["coords"]["lon"],
        },
        name="fields",
    )
    return ds


def get(time: datetime.datetime, channel_set: schema.ChannelSet) -> xarray.DataArray:

    root = config.get_data_root(channel_set)
    path = _get_path(root, time)
    logger.debug(f"Opening {path} for {time}.")

    metadata_path = os.path.join(config.ERA5_HDF5_73, "data.json")
    metadata_path = filesystem.download_cached(metadata_path)
    with open(metadata_path) as mf:
        metadata = json.load(mf)
    ds = _get_hdf5(path=path, metadata=metadata, time=time)
    return ds
