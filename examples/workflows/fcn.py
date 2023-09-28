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
import numpy as np
import datetime
import subprocess

# Set number of GPUs to use to 1
os.environ["WORLD_SIZE"] = "1"
# Set model registry as a local folder
model_registry = os.path.join(os.path.dirname(os.path.realpath(os.getcwd())), "models")
os.makedirs(model_registry, exist_ok=True)
os.environ["MODEL_REGISTRY"] = model_registry

# Download the model checkpoint
if not os.path.isdir(os.path.join(model_registry, "fcn")):
    print("Downloading model checkpoint, this may take a bit")
    subprocess.run(
        [
            "wget",
            "-nc",
            "-P",
            f"{model_registry}",
            "https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_fcn/versions/v0.1/files/fcn.zip",  # noqa
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    subprocess.run(
        ["unzip", "-u", f"{model_registry}/fcn.zip", "-d", f"{model_registry}"]
    )
    subprocess.run(["rm", f"{model_registry}/fcn.zip"])

import earth2mip.networks.fcn as fcn
from earth2mip import registry, inference_ensemble
from earth2mip.initial_conditions import cds
from modulus.distributed import DistributedManager
from os.path import dirname, abspath, join

# %% Load model package and data source
device = DistributedManager().device
print(f"Loading FCN model onto {device}, this can take a bit")
package = registry.get_model("fcn")
sfno_inference_model = fcn.load(package, device=device)

data_source = cds.DataSource(sfno_inference_model.in_channel_names)
output = "path/"
time = datetime.datetime(2018, 1, 1)
ds = inference_ensemble.run_basic_inference(
    sfno_inference_model,
    n=1,
    data_source=data_source,
    time=time,
)

# %% Post-process
import matplotlib.pyplot as plt
from scipy.signal import periodogram

output = f"{dirname(dirname(abspath(__file__)))}/outputs/workflows"
os.makedirs(output, exist_ok=True)

arr = ds.sel(channel="u100m").values
f, pw = periodogram(arr, axis=-1, fs=1)
pw = pw.mean(axis=(1, 2))

l = ds.time - ds.time[0]  # noqa
days = l / (ds.time[-1] - ds.time[0])
cm = plt.cm.get_cmap("viridis")
for k in range(ds.sizes["time"]):
    day = (ds.time[k] - ds.time[0]) / np.timedelta64(1, "D")
    day = day.item()
    plt.loglog(f, pw[k], color=cm(days[k]), label=day)
plt.legend()
plt.ylim(bottom=1e-8)
plt.grid()
plt.savefig(join(output, "u200_spectra_fcn.png"), bbox_inches="tight")

# %%
day = (ds.time - ds.time[0]) / np.timedelta64(1, "D")
plt.semilogy(day, pw[:, 100:].mean(-1), "o-")
plt.savefig(join(output, "u200_high_wave_fcn.png"), bbox_inches="tight")
