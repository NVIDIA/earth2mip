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
import xarray

from earth2mip.datasets.hindcast import open_forecast

__all__ = ["apply", "compute_edges"]


def cleanup_metadata(fct_p):
    fct_p = fct_p.drop("category_edge").rename(lat="latitude", lon="longitude")
    fct_p = fct_p[["t2m"]]
    fct_p = fct_p.sel(lead_time=[14, 28])
    # TODO check why the latitude is off
    fct_p = fct_p.reindex(
        latitude=np.linspace(90, -90, fct_p.sizes["latitude"]),
        longitude=np.linspace(0, 360, fct_p.sizes["longitude"], endpoint=False),
    )
    fct_p["lead_time"] = fct_p.lead_time * np.timedelta64(1, "D")
    translate = np.vectorize(lambda x: "near normal" if x == "normal" else x)
    fct_p["category"] = translate(fct_p.category)
    return fct_p


def apply(path, tercile_edges, output):
    ens = open_forecast(
        path, group="ensemble.zarr", chunks={"initial_time": 1, "ensemble": 1}
    )

    # %% [markdown]
    # Moderately expensive: run time = 90 seconds
    #

    # %%
    tercile = xarray.open_dataset(tercile_edges)
    terciles_as_forecast = tercile.sel(week=ens.initial_time.dt.week)
    terciles_as_forecast

    # %%
    below_normal = (ens < terciles_as_forecast.isel(category_edge=0)).mean("ensemble")
    above_normal = (ens >= terciles_as_forecast.isel(category_edge=1)).mean("ensemble")
    normal = 1 - below_normal - above_normal

    terciled = xarray.concat(
        [below_normal, normal, above_normal],
        dim=xarray.Variable("category", ["below normal", "normal", "above normal"]),
        coords="minimal",
        compat="override",
    )
    print(terciled)
    # rename to match contest metadata
    terciled = terciled.rename(initial_time="forecast_time")
    terciled["lead_time"] = 14 * terciled.lead_time
    terciled = cleanup_metadata(terciled)
    terciled.to_netcdf(output)


def compute_edges(path, output):
    ds = open_forecast(
        path,
        group="ensemble.zarr",
        chunks={"initial_time": -1, "ensemble": -1, "lon": "auto"},
    )
    ds = ds[["t2m"]]

    tercile = (
        ds.groupby(ds.initial_time.dt.week)
        .quantile(q=[1.0 / 3.0, 2.0 / 3.0], dim=["initial_time", "ensemble"])
        .rename({"quantile": "category_edge"})
        .astype("float32")
    )
    tercile = tercile.load()
    tercile.to_netcdf(output)
