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
"""
pip install -r earth2mip/networks/graphcast/requirements.txt


Run like::

    python3 examples/graphcast_simple.py

"""
# %%
import sys

sys.path.insert(0, "..")
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray
from earth2mip.initial_conditions import cds

from earth2mip.model_registry import Package
from earth2mip.networks.graphcast import channels, inference


def get_input_from_xarray(task_config, example_batch):
    arrays = []
    levels = list(task_config.pressure_levels)
    # needs to be compatible with earth2mip.networks.graphcast.inference.get_code
    # TODO remove the possibility of this bug
    state_variables = [
        v for v in task_config.target_variables if v in task_config.input_variables
    ]
    for v in sorted(state_variables):
        if channels.is_3d(v):
            # b, h, p, y, x
            arr = example_batch[v].sel(level=levels).isel(time=slice(0, 2)).values
            assert arr.ndim == 5
            arrays.append(arr)
        else:
            arr = example_batch[v].isel(time=slice(0, 2)).values
            assert arr.ndim == 4
            arr = np.expand_dims(arr, 2)
            arrays.append(arr)

    array = np.concatenate(arrays, axis=2)
    return np.flip(array, axis=-2).copy()


# %%
# https://console.cloud.google.com/storage/browser/dm_graphcast/dataset?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false
root = "gs://dm_graphcast"
package = Package(root, seperator="/")
time_loop = inference.load_time_loop(package, version="operational")

dataset_filename = package.get(
    "dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc"
)
with open(dataset_filename, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

# %%
task_config = time_loop.task_config
target_codes = channels.get_codes(
    task_config.target_variables, task_config.pressure_levels, [0]
)

# TODO use one of the builtin data sources
# array = get_input_from_xarray(task_config, example_batch)

data_source = cds.DataSource(time_loop.in_channel_names)
time = datetime.datetime(2018, 1, 1)
arrays = [
    data_source[time - k * time_loop.history_time_step].sel(
        channel=time_loop.in_channel_names
    )
    for k in range(time_loop.n_history_levels)
]
array = np.concatenate(arrays)
# TODO make the dtype flexible
x = torch.from_numpy(array).cuda().type(torch.float)
# need a batch dimension of length 1
x = x[None]

i = time_loop.out_channel_names.index("tp06")
for k, (time, x, _) in enumerate(time_loop(time, x)):
    print(k)
    plt.pcolormesh(x[0, i].cpu().numpy())
    plt.savefig(f"{k:03d}.png")
    if k == 10:
        break
