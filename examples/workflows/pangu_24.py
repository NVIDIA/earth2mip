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

# Set number of GPUs to use to 1
os.environ["WORLD_SIZE"] = "1"
# Set model registry as a local folder
model_registry = os.path.join(os.path.dirname(os.path.realpath(os.getcwd())), "models")
os.makedirs(model_registry, exist_ok=True)
os.environ["MODEL_REGISTRY"] = model_registry

import earth2mip.networks.pangu as pangu
from earth2mip import (
    registry,
    inference_ensemble,
)
from earth2mip.initial_conditions import cds
from os.path import dirname, abspath, join

# %% Load model package and data source
print("Loading pangu model, this can take a bit")
package = registry.get_model("e2mip://pangu_24")
# Load just the 24 hour model (also supports 6 hour)
inferener = pangu.load_single_model(package, time_step_hours=24)
data_source = cds.DataSource(inferener.in_channel_names)
time = datetime.datetime(2018, 1, 1)

# %% Run inference
ds = inference_ensemble.run_basic_inference(
    inferener,
    n=12,
    data_source=data_source,
    time=time,
)
print(ds)

# %% Post-process
import matplotlib.pyplot as plt
from scipy.signal import periodogram

output = f"{dirname(dirname(abspath(__file__)))}/outputs/workflows"
os.makedirs(output, exist_ok=True)

arr = ds.sel(channel="u200").values
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
plt.savefig(join(output, "u200_spectra_pangu24.png"), bbox_inches="tight")

# %%
day = (ds.time - ds.time[0]) / np.timedelta64(1, "D")
plt.semilogy(day, pw[:, 100:].mean(-1), "o-")
plt.savefig(join(output, "u200_high_wave_pangu24.png"), bbox_inches="tight")
