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

from earth2mip import config
import datetime
from earth2mip import schema, regrid, time_loop
from earth2mip.initial_conditions.era5 import open_era5_xarray, HDF5DataSource
from earth2mip.initial_conditions import ifs, cds, gfs, hrmip, base
import torch

__all__ = ["open_era5_xarray", "get_data_source"]


def get_data_source(
    n_history,
    grid,
    channel_set: schema.ChannelSet,
    netcdf="",
    initial_condition_source=schema.InitialConditionSource.era5,
) -> base.DataSource:
    channel_names = channel_set.list_channels()
    if initial_condition_source == schema.InitialConditionSource.era5:
        return HDF5DataSource.from_path(
            root=config.get_data_root(channel_set), n_history=n_history
        )
    elif initial_condition_source == schema.InitialConditionSource.cds:
        return cds.DataSource(channel_names)
    elif initial_condition_source == schema.InitialConditionSource.gfs:
        return gfs.DataSource(channel_names)
    elif initial_condition_source == schema.InitialConditionSource.ifs:
        return ifs.DataSource(channel_names)
    elif initial_condition_source == schema.InitialConditionSource.hrmip:
        return hrmip.HDFPlSl(path=config.get_data_root(channel_set))
    else:
        raise NotImplementedError(initial_condition_source)


def get_initial_condition_for_model(
    time_loop: time_loop.TimeLoop, data_source: base.DataSource, time: datetime
) -> torch.Tensor:
    array = data_source[time]
    index = [data_source.channel_names.index(c) for c in time_loop.in_channel_names]
    values = array[:, index]
    regridder = regrid.get_regridder(data_source.grid, time_loop.grid).to(
        time_loop.device
    )
    # TODO make the dtype flexible
    x = torch.from_numpy(values).cuda().type(torch.float)
    # need a batch dimension of length 1
    x = x[None]
    x = regridder(x)
    return x
