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
from earth2mip.initial_conditions import cds

from earth2mip.model_registry import Package
from earth2mip.networks.graphcast import inference

# %%
# https://console.cloud.google.com/storage/browser/dm_graphcast/dataset?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false
root = "gs://dm_graphcast"
package = Package(root, seperator="/")
time_loop = inference.load_time_loop(package, version="operational")

# %%
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
