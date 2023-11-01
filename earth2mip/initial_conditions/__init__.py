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

from typing import List
from earth2mip import config
import datetime
from earth2mip import schema, regrid, time_loop
from earth2mip.initial_conditions import ifs, cds, gfs, hrmip, base, hdf5
import numpy as np
import torch

__all__ = ["get_data_source", "cds", "ifs", "gfs", "hrmip", "hdf5"]


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


def get_initial_condition_for_model(
    time_loop: time_loop.TimeLoop, data_source: base.DataSource, time: datetime
) -> torch.Tensor:

    dt = time_loop.history_time_step
    arrays = []
    for i in range(time_loop.n_history_levels - 1, -1, -1):
        time_to_get = time - i * dt
        arr = data_source[time_to_get]
        expected_shape = (len(data_source.channel_names), *data_source.grid.shape)
        arrays.append(arr)
        if arr.shape != expected_shape:
            raise ValueError(time_to_get, arr.shape, expected_shape)

    # stack the history
    array = np.stack(arrays, axis=0)

    # make an empty batch dim
    array = array[None]

    index = [data_source.channel_names.index(c) for c in time_loop.in_channel_names]
    values = array[:, :, index]
    regridder = regrid.get_regridder(data_source.grid, time_loop.grid).to(
        time_loop.device
    )
    # TODO make the dtype flexible
    x = torch.from_numpy(values).to(time_loop.device).type(torch.float)
    # need a batch dimension of length 1
    x = regridder(x)
    return x
