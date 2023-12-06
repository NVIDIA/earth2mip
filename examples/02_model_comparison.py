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
Basic Inference with Multiple Models
=====================================

The following notebook demonstrates how to use Earth-2 MIP for running different AI
weather models and comparing their outputs. Specifically, this will compare the Pangu
weather model and Deep Learning Weather Prediction (DLWP) mode with an intial state
pulled from the Climate Data Store (CDS). This will also how how to interact with
Earth-2 MIP using Python APIs for greater control over inference workflows.

In summary this notebook will cover the following topics:

- Configuring and setting up Pangu Model Registry and DLWP Model Registry
- Setting up a basic deterministic inferencer for both models
- Running inference in a Python script
- Post processing results
"""

# %%
# Set Up
# ------
# Starting off with imports, hopefully you have already installed Earth-2 MIP from this
# repository. See the previous notebook for information about configuring Earth-2 MIP, its
# assumed enviroment variables have already been properly set.


# %%
import datetime
import os

import dotenv
import xarray

dotenv.load_dotenv()

from earth2mip import inference_ensemble, registry
from earth2mip.initial_conditions import cds

# %%
# The cell above created a model registry folder for us, now we need to populate it with
# model packages.
# We will start with Pangu, which is a model that uses ONNX checkpoints.
# Since this is a built in model, we can use the `registry.get_model` function with the
# `e2mip://` prefix to auto download the checkpoints.
# Under the hood, this is fetching the ONNX checkpoints and creating a `metadata.json`
# file to help Earth-2 MIP know how to load the model into memory for inference.

# %%
print("Fetching Pangu model package...")
package = registry.get_model("e2mip://pangu")

# %%
# Next DLWP model package will need to be downloaded. This model follows the standard
# proceedure most do in Earth-2 MIP, being served via Modulus and hosted on NGC model
# registry.

# %%
print("Fetching DLWP model package...")
package = registry.get_model("e2mip://dlwp")

# %%
# The final setup step is to set up your CDS API key so we can access ERA5 data to act as
# an initial state. Earth-2 MIP supports a number of different initial state data sources
# that are supported including HDF5, CDS, GFS, etc. The CDS initial state provides a
# convenient way to access a limited amount of historical weather data. Its recommended
# for accessing an initial state, but larger data requirements should use locally stored
# weather datasets.
#
# Enter your CDS API uid and key below (found under your profile page).
# If you don't a CDS API key, find out more here.
#
# - `https://cds.climate.copernicus.eu/cdsapp#!/home <https://cds.climate.copernicus.eu/cdsapp#!/home>`_
# - `https://cds.climate.copernicus.eu/api-how-to <https://cds.climate.copernicus.eu/api-how-to>`_

# %%
cds_api = os.path.join(os.path.expanduser("~"), ".cdsapirc")
if not os.path.exists(cds_api):
    uid = input("Enter in CDS UID (e.g. 123456): ")
    key = input("Enter your CDS API key (e.g. 12345678-1234-1234-1234-123456123456): ")
    # Write to config file for CDS library
    with open(cds_api, "w") as f:
        f.write("url: https://cds.climate.copernicus.eu/api/v2\n")
        f.write(f"key: {uid}:{key}\n")

# %%
# Running Inference
# -----------------
# To run inference of these models we will use some of Earth-2 MIPs Python APIs to perform
# inference. The first step is to load the model from the model registry, which is done
# using the `registry.get_model` command. This will look in your `MODEL_REGISTRY` folder
# for the provided name and use this as a filesystem for loading necessary files.
#
# The model is then loaded into memory using the load function for that particular
# network. Earth-2 MIP has multiple abstracts that can allow this to be automated that can
# be used instead if desired.

# %%
import earth2mip.networks.dlwp as dlwp
import earth2mip.networks.pangu as pangu

# Output directoy
output_dir = "outputs/02_model_comparison"
os.makedirs(output_dir, exist_ok=True)

print("Loading models into memory")
# Load DLWP model from registry
package = registry.get_model("dlwp")
dlwp_inference_model = dlwp.load(package)

# Load Pangu model(s) from registry
package = registry.get_model("pangu")
pangu_inference_model = pangu.load(package)

# %%
# Next we set up the initial state data source for January 1st, 2018 at 00:00:00 UTC.
# As previously mentioned, we will pull data on the fly from CDS (make sure you set up
# your API key above). Since DLWP and Pangu require different channels (and time steps),
# we will create two seperate data-sources for them.

# %%
time = datetime.datetime(2018, 1, 1)

# DLWP datasource
dlwp_data_source = cds.DataSource(dlwp_inference_model.in_channel_names)

# Pangu datasource, this is much simplier since pangu only uses one timestep as an input
pangu_data_source = cds.DataSource(pangu_inference_model.in_channel_names)

# %%
# With the initial state downloaded for each and set up in an Xarray dataset, we can now
# run deterministic inference for both which can be achieved using the
# `inference_ensemble.run_basic_inference` method which will produce a Xarray
# `data array <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_ to then
# work with.

# %%
print("Running Pangu inference")
pangu_ds = inference_ensemble.run_basic_inference(
    pangu_inference_model,
    n=24,  # Note we run 24 steps here because Pangu is at 6 hour dt (6 day forecast)
    data_source=pangu_data_source,
    time=time,
)
pangu_ds.to_netcdf(f"{output_dir}/pangu_inference_out.nc")
print(pangu_ds)

# %%
print("Running DLWP inference")
dlwp_ds = inference_ensemble.run_basic_inference(
    dlwp_inference_model,
    n=24,  # Note we run 24 steps. DLWP steps at 12 hr dt, but yeilds output every 6 hrs (6 day forecast)
    data_source=dlwp_data_source,
    time=time,
)
dlwp_ds.to_netcdf(f"{output_dir}/dlwp_inference_out.nc")
print(dlwp_ds)

# %%
# Post Processing
# ---------------
# With inference complete, now the fun part: post processing and analysis!
# Here we will just plot the z500 (geopotential at pressure level 500) contour time-series of both models.

# %%
import matplotlib.pyplot as plt

# Open dataset from saved NetCDFs
pangu_ds = xarray.open_dataarray(f"{output_dir}/pangu_inference_out.nc")
dlwp_ds = xarray.open_dataarray(f"{output_dir}/dlwp_inference_out.nc")

# Get data-arrays at 12 hour steps
pangu_arr = pangu_ds.sel(channel="z500").values[::2]
dlwp_arr = dlwp_ds.sel(channel="z500").values[::2]
# Plot
plt.close("all")
fig, axs = plt.subplots(2, 13, figsize=(13 * 4, 5))
for i in range(13):
    axs[0, i].imshow(dlwp_arr[i, 0])
    axs[1, i].imshow(pangu_arr[i, 0])
    axs[0, i].set_title(time + datetime.timedelta(hours=12 * i))

axs[0, 0].set_ylabel("DLWP")
axs[1, 0].set_ylabel("Pangu")
plt.suptitle("z500 DLWP vs Pangu")
plt.savefig(f"{output_dir}/pangu_dlwp_z500.png")

# %%
# And that completes the second notebook detailing how to run deterministic inference of
# two models using Earth-2 MIP. In the next notebook, we will look at how to score a
# model compared against ERA5 re-analysis data.
