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
from earth2mip import (
    registry,
    inference_ensemble,
)
from earth2mip.networks.dlwp import load
from earth2mip.initial_conditions import cds
import xarray as xr

# %%

package = registry.get_model("dlwp")

timeloop = load(package, device="cuda:0")
# data_source = HDF5DataSource.from_path("s3://ERA5_2D_73var", n_history=1)
cds_data_source = cds.DataSource(
    ["t850", "z1000", "z700", "z500", "z300", "tcwv", "t2m"]
)

time = datetime.datetime(2018, 1, 1)
ds1 = cds_data_source[time].sel(
    channel=["t850", "z1000", "z700", "z500", "z300", "tcwv", "t2m"]
)
ds2 = cds_data_source[time - datetime.timedelta(hours=6)].sel(
    channel=["t850", "z1000", "z700", "z500", "z300", "tcwv", "t2m"]
)
ds = xr.concat([ds2, ds1], dim="time")

data_source = {time: ds}
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
# from scipy.signal import periodogram
import matplotlib.pyplot as plt

arr = ds.sel(channel="t2m").values
fig, axs = plt.subplots(1, 13, figsize=(13 * 5, 5))
for i in range(13):
    axs[i].imshow(arr[i, 0])
plt.savefig("t2m_field.png")
