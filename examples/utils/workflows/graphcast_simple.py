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

    pip install -r requirements.txt
    pip install -e .[graphcast]
    python3 examples/workflows/graphcast_simple.py

This emample also demonstrates the use of the functional TimeStepper API

"""
# %%
import sys

sys.path.insert(0, "..")
import datetime
import logging

import matplotlib.pyplot as plt

import earth2mip.networks.graphcast
from earth2mip.initial_conditions import cds, get_initial_condition_for_model
from earth2mip.model_registry import Package

logging.basicConfig(level=logging.INFO)

# %%
# Can review the data in graphcast Google storage bucket here:
# https://console.cloud.google.com/storage/browser/dm_graphcast/dataset?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false
root = "gs://dm_graphcast"
package = Package(root, seperator="/")
time_loop = earth2mip.networks.graphcast.load_time_loop_operational(package)
stepper = time_loop.stepper

# Can also load like this for simplicity:
# from earth2mip.networks import get_model
# time_loop = get_model("e2mip://graphcast_operational")

# %%
time = datetime.datetime(2018, 1, 1)
data_source = cds.DataSource(time_loop.in_channel_names)
x = get_initial_condition_for_model(time_loop, data_source, time)

state = stepper.initialize(x, time)
print("Graphcast's state is a tuple of (datetime, xarray.Dataset, rng)", state)

# %%

field = "tp06"
i = time_loop.out_channel_names.index(field)
for k in range(10):
    print(k)
    state, output = stepper.step(state)
    plt.clf()
    plt.pcolormesh(output[0, i].cpu().numpy())
    plt.colorbar()
    plt.savefig(f"{k:03d}.png")
