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
import dataclasses
from typing import List

import numpy as np

from earth2mip import schema


@dataclasses.dataclass(frozen=True)
class LatLonGrid:

    lat: List[float]
    lon: List[float]

    @property
    def shape(self):
        return (len(self.lat), len(self.lon))


def equiangular_lat_lon_grid(
    nlat: int, nlon: int, includes_south_pole: bool = True
) -> LatLonGrid:
    """A regular lat-lon grid

    Lat is ordered from 90 to -90. Includes -90 and only if if includes_south_pole is True.
    Lon is ordered from 0 to 360. includes 0, but not 360.

    """  # noqa
    lat = np.linspace(90, -90, nlat, endpoint=includes_south_pole)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    return LatLonGrid(lat.tolist(), lon.tolist())


def from_enum(grid_enum: schema.Grid) -> LatLonGrid:
    if grid_enum == schema.Grid.grid_720x1440:
        return equiangular_lat_lon_grid(720, 1440, includes_south_pole=False)
    elif grid_enum == schema.Grid.grid_721x1440:
        return equiangular_lat_lon_grid(721, 1440)
    elif grid_enum == schema.Grid.s2s_challenge:
        return equiangular_lat_lon_grid(181, 360)
    else:
        raise ValueError(f"Unknown grid {grid_enum}")
