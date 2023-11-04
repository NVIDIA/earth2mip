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
# %% [markdown]
"""
# Scoring Models in Earth-2 MIP

The following notebook will demonstrate how to use Earth-2 MIP to perform a scoring
workflow to assess the accuracy of AI models using ERA5 reanalysis data as the
ground truth. This can then be extended to score custom models placed into the model
registry. This tutorial also covers details about using a HDF5 datasource, the expected
format this data should be formatted and how to use H5 files for evaluating models over
a year's worth of data.

In summary this notebook will cover the following topics:

- Implementing a basic scoring workflow in Earth-2 MIP
- HDF5 datasource and the expected data format of the H5 files

"""
# %%
import os
import xarray
import datetime
import numpy as np

# %% [markdown]
"""
## Setting up HDF5 data

The first step of scoring is handling the target data. One could simply use the
CDSDatasource to download target data on the fly, but depending on how comprehensive
the scoring is, this can prove quite slow.
Additionally, many scoring pipelines require on-prem data.
Thus, this will demonstrate how to use the HDF5 datasource.

The HDF5 data source assumes that the data to be loaded is stored in the general form:

year.h5
 | - field (time, channels, grid)

For AFNO which requires 34 channels with a time-step size of 6 hours, an H5 file will
have the following form of data:

2017.h5
 | - field (1460, 34, 720, 1440)
2016.h5
 | - field (1464, 34, 720, 1440)
2015.h5
 | - field (1460, 34, 720, 1440)

(Note the later two dims have some flexibility with regridding)

One option to build these H5 files from scratch is to use the ERA5 mirror scripts
provided in [Modulus](https://github.com/NVIDIA/modulus/tree/main/examples/weather/dataset_download).
For the rest of this tutorial, it is assumed that 2017.h5 is present for the full year.
"""  # noqa: E501

# %%
h5_folder = "/mount/34vars/test/"

# %% [markdown]
"""
## HDF5 Datasource

With the H5 files properly formatted, a H5 datasource can now get defined.
This requires two items: a root directory location of the H5 files as well as some
metadata.
The metadata is a JSON/dictionary object that helps Earth-2 MIP index the H5 file.
Typically, this can be done by placing a `data.json` file next to the H5 files,
Pythonically metadata for a 34 channel H5 files described above is given below.
"""
# %%
from earth2mip.initial_conditions import hdf5

metadata = {
    "h5_path": "fields",
    "attrs": {"decription": "Custom HDF5 data"},
    "dims": ["time", "channel", "lat", "lon"],
    "dhours": 6,
    "coords": {
        "lat": np.linspace(90, -90, 721),
        "lon": np.linspace(0, 359.75, 1440),
        "channel": [
            "u10m",
            "v10m",
            "t2m",
            "sp",
            "msl",
            "t850",
            "u1000",
            "v1000",
            "z1000",
            "u850",
            "v850",
            "z850",
            "u500",
            "v500",
            "z500",
            "t500",
            "z50",
            "r500",
            "r850",
            "tcwv",
            "u100m",
            "v100m",
            "u250",
            "v250",
            "z250",
            "t250",
            "u100",
            "v100",
            "z100",
            "t100",
            "u900",
            "v900",
            "z900",
            "t900",
        ],
    },
}

datasource = hdf5.DataSource(root=h5_folder, metadata=metadata)

# Test to see if our datasource is working
time = datetime.datetime(2017, 5, 1, 18)
out = datasource[time]

# %% [markdown]
"""
## Loading Models

With the HDF5 datasource loaded the next step is to load our model we wish to score.
In this tutorial we will be using the built in FourcastNet model.
Take note of the `e2mip://` which will direct Earth-2 MIP to load a known model package.
FourcastNet is selected here simply because its a 34 channel model which aligns with the
H5 files described above.
"""
# %%
import earth2mip.networks.fcn as fcn
from earth2mip import registry
from modulus.distributed import DistributedManager

device = DistributedManager().device
package = registry.get_model("e2mip://fcn")
model = fcn.load(package, device=device)

# %% [markdown]
"""
## Running Scoring

With the datasource and model loaded, scoring can now be performed.
To score this we will run 10 day forecasts over the span of the entire year at 30 day
intervals.
For research, one would typically want this to be much more comprehensive so feel free
to customize for you're use case.

The `score_deterministic` API provides a simple way to calculate RMSE and ACC scores.
ACC scores require climatology which is beyond the scope of this example, thus zero
values will be provided and only the RMSE will be of concern.
"""

# %%
from earth2mip.inference_medium_range import score_deterministic

time = datetime.datetime(2017, 1, 1, 0)
initial_times = [time + datetime.timedelta(days=30 * i) for i in range(12)]

if not os.path.exists("scoring_output.nc"):
    output = score_deterministic(
        model,
        n=40,  # 6 hour timesteps
        initial_times=initial_times,
        data_source=datasource,
        time_mean=np.zeros((34, 721, 1440)),
    )
    output.to_netcdf("scoring_output.nc")
    print(output)

# %% [markdown]
"""
## Post Processing

The last step is any post processing / IO that is desired.
Typically its recommended to save the output dataset to a netCDF file for further
processing.
Lets plot the RMSE of the z500 field.
"""

# %%
import matplotlib.pyplot as plt

dataset = xarray.open_dataset("scoring_output.nc")

plt.close("all")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dataset.rmse.sel(channel="z500").values)

# Plot clean up
ticks = np.arange(0, 41, 5)
ax.set_xticks(ticks)
ax.set_xticklabels([f"{6*i}" for i in ticks])
ax.set_xlabel("Lead Time")
ax.set_ylabel("RMSE")
ax.set_title("FourcastNet z500 RMSE 2017")

plt.draw()

# %%
# Clean up
os.remove("scoring_output.nc")
