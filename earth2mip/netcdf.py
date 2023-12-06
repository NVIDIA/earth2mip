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

"""Routines to save domains to a netCDF file
"""
from typing import Iterable, List

import numpy as np
import torch
import xarray as xr

import earth2mip.grid
from earth2mip import geometry
from earth2mip.diagnostics import Diagnostics, DiagnosticTypes
from earth2mip.weather_events import Domain

__all__ = ["initialize_netcdf", "update_netcdf"]


def _assign_lat_attributes(nc_variable):
    nc_variable.units = "degrees_north"
    nc_variable.standard_name = "latitude"
    nc_variable.long_name = "latitude"


def _assign_lon_attributes(nc_variable):
    nc_variable.units = "degrees_east"
    nc_variable.standard_name = "longitude"
    nc_variable.long_name = "longitude"


def init_dimensions(domain: Domain, group, grid: earth2mip.grid.LatLonGrid):

    lat = np.array(grid.lat)
    lon = np.array(grid.lon)

    if domain.type == "CWBDomain":
        cwb_path = "/lustre/fsw/sw_climate_fno/nbrenowitz/2023-01-24-cwb-4years.zarr"
        lat = xr.open_zarr(cwb_path)["XLAT"][:, 0]
        lon = xr.open_zarr(cwb_path)["XLONG"][0, :]
        nlat = lat.size
        nlon = lon.size
        group.createDimension("lat", nlat)
        group.createDimension("lon", nlon)
        v = group.createVariable("lat", np.float32, ("lat"))
        _assign_lat_attributes(v)
        v = group.createVariable("lon", np.float32, ("lon"))
        _assign_lon_attributes(v)

        group["lat"][:] = lat
        group["lon"][:] = lon
    elif domain.type == "Window":
        lat_sl, lon_sl = geometry.get_bounds_window(domain, lat, lon)
        group.createVariable("imin", int, ())
        group.createVariable("imax", int, ())
        group.createVariable("jmin", int, ())
        group.createVariable("jmax", int, ())

        group["imin"][:] = lat_sl.start
        group["imax"][:] = lat_sl.stop
        group["jmin"][:] = lon_sl.start
        group["jmax"][:] = lon_sl.stop

        nlat = np.r_[lat_sl].size
        nlon = np.r_[lon_sl].size
        group.createDimension("lat", nlat)
        group.createDimension("lon", nlon)
        v = group.createVariable("lat", np.float32, ("lat"))
        _assign_lat_attributes(v)
        v = group.createVariable("lon", np.float32, ("lon"))
        _assign_lon_attributes(v)

        group["lat"][:] = lat[lat_sl]
        group["lon"][:] = lon[lon_sl]

    elif domain.type == "MultiPoint":
        assert len(domain.lat) == len(  # noqa
            domain.lon
        ), "Lat and Lon arrays must be of same size!"
        group.createDimension("npoints", len(domain.lon))
        v = group.createVariable("lat_point", np.float32, ("npoints"))
        _assign_lat_attributes(v)

        v = group.createVariable("lon_point", np.float32, ("npoints"))
        _assign_lon_attributes(v)

        for diagnostic in domain.diagnostics:
            group.createDimension("n_channel", len(diagnostic.channels))
        group["lat_point"][:] = domain.lat
        group["lon_point"][:] = domain.lon
    else:
        raise NotImplementedError(f"domain type {domain.type} not supported")
    return


def initialize_netcdf(
    nc, domains: Iterable[Domain], grid: earth2mip.grid.LatLonGrid, n_ensemble, device
) -> List[List[Diagnostics]]:
    nc.createVLType(str, "vls")
    nc.createDimension("time", None)
    nc.createDimension("ensemble", n_ensemble)
    nc.createVariable("time", np.float32, ("time"))
    total_diagnostics = []
    for domain in domains:
        group = nc.createGroup(domain.name)
        init_dimensions(domain, group, grid)
        diagnostics = []
        for d in domain.diagnostics:
            lat = np.array(grid.lat)
            lon = np.array(grid.lon)
            diagnostic = DiagnosticTypes[d.type](
                group, domain, grid, d, lat, lon, device
            )
            diagnostics.append(diagnostic)

        total_diagnostics.append(diagnostics)
    return total_diagnostics


def update_netcdf(
    data: torch.Tensor,
    total_diagnostics: List[List[Diagnostics]],
    domains: List[Domain],
    batch_id,
    time_count,
    grid: earth2mip.grid.LatLonGrid,
    channel_names_of_data: List[str],
):
    assert len(total_diagnostics) == len(domains), (total_diagnostics, domains)  # noqa
    lat = np.array(grid.lat)
    lon = np.array(grid.lon)

    batch_size = geometry.get_batch_size(data)
    for d_index, domain in enumerate(domains):
        lat, lon, regional_data = geometry.select_space(data, lat, lon, domain)

        domain_diagnostics = total_diagnostics[d_index]
        for diagnostic in domain_diagnostics:
            index = [
                channel_names_of_data.index(c) for c in diagnostic.diagnostic.channels
            ]
            output = data[:, index]
            diagnostic.update(output, time_count, batch_id, batch_size)
    return
