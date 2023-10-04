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
import os
import datetime
import xarray as xr
import subprocess

# Set number of GPUs to use to 1
os.environ["WORLD_SIZE"] = "1"
# Set model registry as a local folder
model_registry = os.path.join(os.path.dirname(os.path.realpath(os.getcwd())), "models")
os.makedirs(model_registry, exist_ok=True)
os.environ["MODEL_REGISTRY"] = model_registry

if not os.path.isdir(os.path.join(model_registry, "dlwp")):
    print("Downloading model checkpoint, this may take a bit")
    subprocess.run(
        [
            "wget",
            "-nc",
            "-P",
            f"{model_registry}",
            "https://api.ngc.nvidia.com/v2/models/nvidia/modulus/"
            + "modulus_dlwp_cubesphere/versions/v0.2/files/dlwp_cubesphere.zip",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        [
            "unzip",
            "-u",
            f"{model_registry}/dlwp_cubesphere.zip",
            "-d",
            f"{model_registry}",
        ]
    )
    subprocess.run(["rm", f"{model_registry}/dlwp_cubesphere.zip"])

import earth2mip.networks.dlwp as dlwp
from earth2mip import (
    registry,
    inference_ensemble,
)
from earth2mip.initial_conditions import cds
from modulus.distributed import DistributedManager
from os.path import dirname, abspath, join

# %% Load model package and data source
device = DistributedManager().device
print(f"Loading dlwp model onto {device}, this can take a bit")
package = registry.get_model("dlwp")
inferencer = dlwp.load(package, device=device)
cds_data_source = cds.DataSource(inferencer.in_channel_names)
# Stack two data-sources together for double timestep inputs
time = datetime.datetime(2018, 1, 1)
ds1 = cds_data_source[time]
ds2 = cds_data_source[time - datetime.timedelta(hours=6)]
ds = xr.concat([ds2, ds1], dim="time")
data_source = {time: ds}
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

output = f"{dirname(dirname(abspath(__file__)))}/outputs/workflows"
os.makedirs(output, exist_ok=True)

arr = ds.sel(channel="t2m").values
fig, axs = plt.subplots(1, 13, figsize=(13 * 5, 5))
for i in range(13):
    axs[i].imshow(arr[i, 0])
plt.savefig(join(output, "t2m_field_dlwp.png"), bbox_inches="tight")
