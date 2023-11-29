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

# %%
import datetime
import os
from os.path import join

from modulus.distributed import DistributedManager

import earth2mip.networks.dlwp as dlwp
from earth2mip import (
    inference_ensemble,
    registry,
)
from earth2mip.initial_conditions import cds

# %% Load model package and data source
device = DistributedManager().device
print(f"Loading dlwp model onto {device}, this can take a bit")
package = registry.get_model("e2mip://dlwp")
inferencer = dlwp.load(package, device=device)
data_source = cds.DataSource(inferencer.in_channel_names)
time = datetime.datetime(2018, 1, 1)

# %% Run inference
ds = inference_ensemble.run_basic_inference(
    inferencer,
    n=12,
    data_source=data_source,
    time=time,
)
print(ds)

# %% Post-process
# %% Post-process
import matplotlib.pyplot as plt

output = "outputs"
os.makedirs(output, exist_ok=True)

arr = ds.sel(channel="t2m").values
fig, axs = plt.subplots(1, 13, figsize=(13 * 5, 5))
for i in range(13):
    axs[i].imshow(arr[i, 0])
plt.savefig(join(output, "t2m_field_dlwp.png"), bbox_inches="tight")
