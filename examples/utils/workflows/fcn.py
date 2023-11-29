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

import numpy as np
from modulus.distributed import DistributedManager

import earth2mip.networks.fcn as fcn
from earth2mip import inference_ensemble, registry
from earth2mip.initial_conditions import cds

# %% Load model package and data source
device = DistributedManager().device
print(f"Loading FCN model onto {device}, this can take a bit")
package = registry.get_model("e2mip://fcn")
afno_inference_model = fcn.load(package, device=device)

# Use IC method to get data source, this will regrid the data if needed
time = datetime.datetime(2018, 1, 1)
data_source = cds.DataSource(channel_names=afno_inference_model.in_channel_names)

ds = inference_ensemble.run_basic_inference(
    afno_inference_model,
    n=1,
    data_source=data_source,
    time=time,
)

# %% Post-process
import matplotlib.pyplot as plt
from scipy.signal import periodogram

output = "outputs"
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
