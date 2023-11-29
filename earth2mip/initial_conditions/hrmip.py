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
from typing import List

import h5py
import numpy as np

import earth2mip.grid
from earth2mip import config, filesystem
from earth2mip.initial_conditions.base import DataSource

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


def _get_hdf5(path: str, metadata, time: datetime.datetime) -> np.ndarray:
    dims = metadata["dims"]
    h5_path = metadata["h5_path"]
    variables: List[str] = []
    ic = _get_time(time)
    with h5py.File(path, "r") as f:
        for nm in h5_path:
            if nm == "pl":
                pl = f[nm][ic : ic + 1]
            elif nm == "sl":
                sl = f[nm][ic : ic + 1]

    assert "pl" in locals() and "sl" in locals()  # noqa

    pl_list: List[str] = [pl[:, var_idx] for var_idx in range(pl.shape[1])]
    pl = np.concatenate(pl_list, axis=1)  # pressure level vars flattened
    data = np.concatenate([pl, sl], axis=1)
    return data


class HDFPlSl(DataSource):
    def __init__(self, path: str) -> None:
        self.path = path
        metadata_path = os.path.join(config.ERA5_HDF5_73, "data.json")
        metadata_path = filesystem.download_cached(metadata_path)
        with open(metadata_path) as mf:
            self.metadata = json.load(mf)

    @property
    def grid(self) -> earth2mip.grid.LatLonGrid:
        return earth2mip.grid.equiangular_lat_lon_grid(721, 1440)

    @property
    def channel_names(self) -> List[str]:
        return self.metadata["coords"]["channel"]

    def __getitem__(self, time: datetime.datetime) -> np.ndarray:
        path = _get_path(self.path, time)
        logger.debug(f"Opening {path} for {time}.")
        return _get_hdf5(path=self.path, metadata=self.metadata, time=time)
