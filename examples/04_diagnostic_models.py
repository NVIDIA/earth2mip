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
"""
Diagnostic Models for Precipitation
===================================

The following notebook will demonstrate how to use diagnostic models inside of Earth-2
MIP for transforming outputs of global weather models into different quantities of
interest. More information on diagnostics can be found in the `user guide <https://nvidia.github.io/earth2mip/userguide/diagnostic.html>`_.

In summary this notebook will cover the following topics:

- Loading a built in diagnostic model for predicting total precipitation
- Combining the diagnostic model with a prognostic model using the DiangosticLooper

"""
# %%
import datetime
import os
import dotenv

dotenv.load_dotenv()

# %%
# Loading Diagnostic Models
# -------------------------
# Loading diagnostic models is similar to prognostic models, but presently use a
# slightly different API. In this example we will using the built in AFNO FourCast Net
# to serve as the underlying prognostic model that will drive the time-ingration. The
# :code:`PrecipitationAFNO` model will then be used to "post-process" the outputs of
# this model to predict precipitation. The key API to load a diagnostic model is the
# :code:`load_diagnostic(package)` function which takes a model package in. If you're
# interested in using the built in model package (i.e. checkpoint), then the
# :code:`load_package()` function can do this for you.

# %%
from modulus.distributed.manager import DistributedManager
from earth2mip.networks import get_model
from earth2mip.diagnostic import PrecipitationAFNO

device = DistributedManager().device

print("Loading FCN model")
model = get_model("e2mip://fcn", device=device)

print("Loading precipitation model")
package = PrecipitationAFNO.load_package()
diagnostic = PrecipitationAFNO.load_diagnostic(package)

# %%
# The next step is to wrap the prognostic model with the Diagnostic Time loop.
# Essentially this adds the execution of the diagnostic model on top of the forecast
# model iterator. This will add the total preciptation field (`tp`) to the output data
# which can the be further processed.

# %%
from earth2mip.diagnostic import DiagnosticTimeLoop

model_diagnostic = DiagnosticTimeLoop(diagnostics=[diagnostic], model=model)

# %%
# Running Inference
# -----------------
# With the diagnostic time loop created the final steps are to create the data source
# and run inference. For this example we will use the CDS data source again. Its assumed
# your CDS API key is already set up. Reference the `first example <https://nvidia.github.io/earth2mip/examples/01_ensemble_inference.html#set-up>`_
# for additional information. We will use the basic inference workflow which returns a
# Xarray dataset we will save to netCDF.

# %%
from earth2mip.inference_ensemble import run_basic_inference
from earth2mip.initial_conditions import cds

print("Constructing initializer data source")
data_source = cds.DataSource(model.in_channel_names)
time = datetime.datetime(2018, 4, 4)

print("Running inference")
output_dir = "outputs/04_diagnostic_precip"
os.makedirs(output_dir, exist_ok=True)
ds = run_basic_inference(
    model_diagnostic,
    n=20,
    data_source=data_source,
    time=time,
)
ds.to_netcdf(os.path.join(output_dir, "precipitation_afno.nc"))
print(ds)

# %%
# Post Processing
# ---------------
# With inference complete we can do some post processing on our predictions. Lets first
# visualize the total precipitation and total column water vapor for a few days.

# %%
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


plt.close("all")
# Open dataset from saved NetCDFs
ds = xr.open_dataarray(os.path.join(output_dir, "precipitation_afno.nc"))

ndays = 3
proj = ccrs.Robinson()
fig, ax = plt.subplots(
    2,
    ndays,
    figsize=(15, 5),
    subplot_kw={"projection": proj},
    gridspec_kw={"wspace": 0.05, "hspace": 0.007},
)

for day in range(ndays):
    i = 4 * day  # 6-hour timesteps
    tp = ds[i, 0].sel(channel="tp")
    img = ax[0, day].pcolormesh(
        tp.lon,
        tp.lat,
        tp.values,
        transform=ccrs.PlateCarree(),
        cmap="cividis",
        vmin=0,
        vmax=0.05,
    )
    ax[0, day].set_title(pd.to_datetime(ds.coords["time"])[i])
    ax[0, day].coastlines(color="k")
    plt.colorbar(img, ax=ax[0, day], shrink=0.40)

    tcwv = ds[i, 0].sel(channel="tcwv")
    img = ax[1, day].pcolormesh(
        tcwv.lon,
        tcwv.lat,
        tcwv.values,
        transform=ccrs.PlateCarree(),
        cmap="gist_ncar",
        vmin=0,
        vmax=75,
    )
    ax[1, day].coastlines(resolution="auto", color="k")
    plt.colorbar(img, ax=ax[1, day], shrink=0.40)

ax[0, 0].text(
    -0.07,
    0.55,
    "Total Precipitation (m)",
    va="bottom",
    ha="center",
    rotation="vertical",
    rotation_mode="anchor",
    transform=ax[0, 0].transAxes,
)

ax[1, 0].text(
    -0.07,
    0.55,
    "Total Column\nWater Vapor (kg m-2)",
    va="bottom",
    ha="center",
    rotation="vertical",
    rotation_mode="anchor",
    transform=ax[1, 0].transAxes,
)

plt.savefig(f"{output_dir}/diagnostic_tp_tcwv.png")


# %%
# This partiulcar date was selected for inference due to an atmopsheric river occuring
# over the west coast of the United States. Lets plot the total precipitation that
# occured over San Francisco.

# %%
plt.close("all")
# Open dataset from saved NetCDFs
ds = xr.open_dataarray(os.path.join(output_dir, "precipitation_afno.nc"))

tp_sf = ds.sel(channel="tp", lat=37.75, lon=57.5)  # Lon is [0, 360]

plt.plot(pd.to_datetime(tp_sf.coords["time"]), tp_sf.values)
plt.title("SF (lat: 37.75N lon: 122.5W)")
plt.ylabel("Total Precipitation (m)")
plt.savefig(f"{output_dir}/sf_tp.png")

# %%
# The land fall of the atmosphric river is very clear here, lets have a look at the
# regional contour of the bay area to better understand the structure of this event.

# %%
plt.close("all")
# Open dataset from saved NetCDFs
ds = xr.open_dataarray(os.path.join(output_dir, "precipitation_afno.nc"))
nsteps = 5
proj = ccrs.AlbersEqualArea(central_latitude=37.75, central_longitude=-122.5)
fig, ax = plt.subplots(
    1,
    nsteps,
    figsize=(20, 5),
    subplot_kw={"projection": proj},
    gridspec_kw={"wspace": 0.05, "hspace": 0.007},
)

for step in range(nsteps):
    i = step + 3
    tp = ds[i, 0].sel(channel="tp")

    ax[step].add_feature(cartopy.feature.OCEAN, zorder=0)
    ax[step].add_feature(cartopy.feature.LAND, zorder=0)
    masked_data = np.ma.masked_where(tp.values < 0.001, tp.values)
    img = ax[step].imshow(
        1000 * masked_data,
        transform=ccrs.PlateCarree(),
        cmap="jet",
        vmin=0,
        vmax=10,
    )
    ax[step].set_title(pd.to_datetime(ds.coords["time"])[i])
    ax[step].coastlines(color="k")
    ax[step].set_extent([-115, -135, 30, 45], ccrs.PlateCarree())
    plt.colorbar(img, ax=ax[step], shrink=0.40)


ax[0].text(
    -0.07,
    0.55,
    "Total Precipitation (mm)",
    va="bottom",
    ha="center",
    rotation="vertical",
    rotation_mode="anchor",
    transform=ax[0].transAxes,
)

plt.savefig(f"{output_dir}/diagnostic_bay_area_tp.png")


# %%
# This completes the introductory notebook on running diagnostic models. Diangostic
# models are signifcantly more cheap to train and more flexible for difference usecases.
# In later examples, we will explore using these models of various other tasks.
