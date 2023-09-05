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
import numpy as np
import datetime
from earth2mip import (
    registry,
    inference_ensemble,
)
from earth2mip.networks.pangu import load_24substep6
from earth2mip.initial_conditions import cds

# %%

package = registry.get_model("pangu_weather_24substep6")

timeloop = load_24substep6(package, device="cuda:0")
# path = sys.argv[1]
# data_source = HDF5DataSource.from_path(path)
data_source = cds.DataSource(timeloop.in_channel_names)
output = "path/"
time = datetime.datetime(2018, 1, 1)
ds = inference_ensemble.run_basic_inference(
    timeloop,
    n=12,
    data_source=data_source,
    time=time,
)
print(ds)


# %%
from scipy.signal import periodogram

arr = ds.sel(channel="u200").values
f, pw = periodogram(arr, axis=-1, fs=1)
pw = pw.mean(axis=(1, 2))
import matplotlib.pyplot as plt

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
plt.savefig("u200_spectra.png")

# %%
day = (ds.time - ds.time[0]) / np.timedelta64(1, "D")
plt.semilogy(day, pw[:, 100:].mean(-1), "o-")
plt.savefig("u200_high_wave.png")
