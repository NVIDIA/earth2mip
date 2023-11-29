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

from typing import Union

import numpy as np
import torch
from netCDF4._netCDF4 import Group

from earth2mip import weather_events
from earth2mip.schema import Grid
from earth2mip.weather_events import CWBDomain, MultiPoint, Window


class Diagnostics:
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        self.group, self.domain, self.grid, self.lat, self.lon = (
            group,
            domain,
            grid,
            lat,
            lon,
        )
        self.diagnostic = diagnostic
        self.device = device

        self._init_subgroup()
        self._init_dimensions()
        self._init_variables()

    def _init_subgroup(
        self,
    ):
        if self.diagnostic.type == "raw":
            self.subgroup = self.group
        else:
            self.subgroup = self.group.createGroup(self.diagnostic.type)

    def _init_dimensions(
        self,
    ):
        if self.domain.type == "MultiPoint":
            self.domain_dims = ("npoints",)
        else:
            self.domain_dims = ("lat", "lon")

    def _init_variables(
        self,
    ):
        dims = self.get_dimensions()
        dtypes = self.get_dtype()
        for channel in self.diagnostic.channels:
            if self.diagnostic.type == "histogram":
                pass
            else:
                self.subgroup.createVariable(
                    channel, dtypes[self.diagnostic.type], dims[self.diagnostic.type]
                )

    def get_dimensions(
        self,
    ):
        raise NotImplementedError

    def get_dtype(
        self,
    ):
        raise NotImplementedError

    def get_variables(
        self,
    ):
        raise NotImplementedError

    def update(
        self,
    ):
        raise NotImplementedError


class Raw(Diagnostics):
    def __init__(
        self,
        group: Group,
        domain: Union[CWBDomain, Window, MultiPoint],
        grid: Grid,
        diagnostic: weather_events.Diagnostic,
        lat: np.ndarray,
        lon: np.ndarray,
        device: torch.device,
    ):
        super().__init__(group, domain, grid, diagnostic, lat, lon, device)

    def get_dimensions(self):
        return {"raw": ("ensemble", "time") + self.domain_dims}

    def get_dtype(self):
        return {"raw": float}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        for c, channel in enumerate(self.diagnostic.channels):
            self.subgroup[channel][batch_id : batch_id + batch_size, time_index] = (
                output[:, c].cpu().numpy()
            )


DiagnosticTypes = {
    "raw": Raw,
}
