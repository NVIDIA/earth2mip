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

"""Routines for working with geometry"""
import numpy as np
import torch

LAT_AVERAGE = "LatitudeAverage"


def get_batch_size(data):
    return data.shape[0]


def get_bounds_window(geom, lat, lon):
    i_min = np.where(lat <= geom.lat_max)[0][0]
    i_max = np.where(lat >= geom.lat_min)[0][-1]
    j_min = np.where(lon >= geom.lon_min)[0][0]
    j_max = np.where(lon <= geom.lon_max)[0][-1]
    return slice(i_min, i_max + 1), slice(j_min, j_max + 1)


def select_space(data, lat, lon, domain):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    assert data.ndim == 4, data.ndim  # noqa
    assert data.shape[2] == lat.size, lat.size  # noqa
    assert data.shape[3] == lon.size, lon.size  # noqa
    domain_type = domain.type
    if domain_type == "Window" or domain_type == LAT_AVERAGE or domain_type == "global":
        lat_sl, lon_sl = get_bounds_window(domain, lat, lon)
        domain_lat = lat[lat_sl]
        domain_lon = lon[lon_sl]
        return domain_lat, domain_lon, data[:, :, lat_sl, lon_sl]
    elif domain_type == "MultiPoint":
        # Convert lat-long points to array index (just got to closest 0.25 degree)
        i = lat.size - np.searchsorted(lat[::-1], domain.lat, side="right")
        j = np.searchsorted(lon, domain.lon, side="left")
        # TODO refactor this assertion to a test
        np.testing.assert_array_equal(domain.lat, lat[i])
        np.testing.assert_array_equal(domain.lon, lon[j])
        return lat[i], lon[j], data[:, :, i, j]
    else:
        raise ValueError(
            f"domain {domain_type} is not supported. Check the weather_events.json"
        )


def bilinear(data: torch.tensor, dims, source_coords, target_coords):
    return
