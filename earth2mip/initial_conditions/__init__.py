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
from typing import List

import numpy as np
import torch

from earth2mip import config, regrid, schema, time_loop
from earth2mip.initial_conditions import base, cds, gfs, hdf5, hrmip, ifs

__all__ = [
    "get_data_source",
    "cds",
    "ifs",
    "gfs",
    "hrmip",
    "hdf5",
]


def get_data_source(
    channel_names: List[str],
    netcdf="",
    initial_condition_source=schema.InitialConditionSource.era5,
) -> base.DataSource:
    if initial_condition_source == schema.InitialConditionSource.era5:
        return hdf5.DataSource.from_path(
            root=config.ERA5_HDF5, channel_names=channel_names
        )
    elif initial_condition_source == schema.InitialConditionSource.cds:
        return cds.DataSource(channel_names)
    elif initial_condition_source == schema.InitialConditionSource.gfs:
        return gfs.DataSource(channel_names)
    elif initial_condition_source == schema.InitialConditionSource.ifs:
        return ifs.DataSource(channel_names)
    elif initial_condition_source == schema.InitialConditionSource.hrmip:
        return hrmip.HDFPlSl(path=config.ERA5_HDF5)
    else:
        raise NotImplementedError(initial_condition_source)


def get_data_from_source(
    data_source: base.DataSource,
    time: datetime.datetime,
    channel_names: List[str],
    grid: schema.Grid,
    n_history_levels: int,
    time_step: datetime.timedelta = datetime.timedelta(hours=0),
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Get data from a data source

    Select the ``channel_names`` and ``regrid`` it to the target ``grid``

    Note:
        Internal helper for scoring routines...not recommended for general use.

    Args:
        time_step: the time step of the time levels
        n_history_levels: the number of history levels to get, see
            :py:class:`earth2mip.time_loop.TimeLoop` for more info.

    Returns:
        (time_levels, c, lat, lon) shaped data
    """
    dt = time_step
    arrays = []
    for i in range(n_history_levels - 1, -1, -1):
        time_to_get = time - i * dt
        arr = data_source[time_to_get]
        expected_shape = (len(data_source.channel_names), *data_source.grid.shape)
        arrays.append(arr)
        if arr.shape != expected_shape:
            raise ValueError(time_to_get, arr.shape, expected_shape)

    # stack the history
    array = np.stack(arrays, axis=0)

    index = [data_source.channel_names.index(c) for c in channel_names]
    values = np.take(array, index, axis=1)
    regridder = regrid.get_regridder(data_source.grid, grid).to(device)
    x = torch.from_numpy(values).to(device).type(dtype)
    # need a batch dimension of length 1
    # make an empty batch dim
    x = x[None]
    x = regridder(x)
    return x


def get_initial_condition_for_model(
    time_loop: time_loop.TimeLoop, data_source: base.DataSource, time: datetime
) -> torch.Tensor:
    return get_data_from_source(
        data_source,
        time,
        time_loop.in_channel_names,
        time_loop.grid,
        time_loop.n_history_levels,
        time_loop.history_time_step,
        time_loop.device,
        time_loop.dtype,
    )
