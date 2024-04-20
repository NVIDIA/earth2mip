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

from earth2mip._units import units
from earth2mip._long_names import long_names
import xarray


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
                var = self.subgroup.createVariable(
                    channel, dtypes[self.diagnostic.type], dims[self.diagnostic.type]
                )
                var.setncattr('units', units[channel])
                var.setncattr('long_name', long_names[channel])

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
        return {"raw": np.float32}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        for c, channel in enumerate(self.diagnostic.channels):
            self.subgroup[channel][batch_id : batch_id + batch_size, time_index] = (
                output[:, c].cpu().numpy()
            )

class IVT(Diagnostics):
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
        #IMPORTANT: these variables must be sorted by pressure level, 
        #from lowest hPa to highest hPa
        self.diagnostic.channels = ["u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", 
                                    "u700", "u850", "u925", "u1000", "v50", "v100", "v150", "v200", "v250", 
                                    "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000", 
                                    "q50", "q100", "q150", "q200", "q250", "q300", "q400", "q500", "q600", "q700", "q850", "q925", "q1000"]
        pressure_levs = []
        for idx, channel in enumerate(self.diagnostic.channels):
            #all the pressure variables have names like r1000
            #where all characters, except the first, are digits
            #the sfc variables are named things like t2m, and
            #they don't have this property

            pressure_var_bool = channel[1:].isdigit()
            if channel[0] == 'u' and pressure_var_bool:
                pressure_levs.append(int(channel[1:]))
        
        pressure_levs = torch.tensor(pressure_levs, device=device)
        #in the variable names (e.g. q850) pressure levels are in hPa, convert to Pa
        self.p = pressure_levs[None,:,None,None].to(device) * 100

    def _init_variables(
        self,
    ):
        dims = self.get_dimensions()
        dtypes = self.get_dtype()
        var = self.subgroup.createVariable(
            "ivt", dtypes[self.diagnostic.type], dims[self.diagnostic.type]
        )
        var.setncattr('units', units["ivt"])
        var.setncattr('long_name', long_names["ivt"])

    def _init_subgroup(
        self,
    ):
        self.subgroup = self.group

    def get_dimensions(self):
        return {"ivt": ("ensemble", "time") + self.domain_dims}

    def get_dtype(self):
        return {"ivt": np.float32}

    def select_output(self, output, channel_name):
        index_to_select = []
        for idx, channel in enumerate(self.diagnostic.channels):
            #all the pressure variables have names like q1000
            #where all characters, except the first, are digits
            #the sfc variables are named things like t2m, and
            #they don't have this property
            pressure_var_bool = channel[1:].isdigit()
            if channel[0] == channel_name and pressure_var_bool:
                index_to_select.append(idx)

        return output[:, index_to_select]
        
    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):

        u = self.select_output(output, "u")
        v = self.select_output(output, "v")
        q = self.select_output(output, "q")

        ivt_u = 1/9.8 * torch.trapz(u*q, self.p, dim=1)
        ivt_v = 1/9.8 * torch.trapz(v*q, self.p, dim=1)
        ivt = torch.sqrt(ivt_u**2 + ivt_v**2)
        self.subgroup["ivt"][
            batch_id : batch_id + batch_size, time_index
        ] = ivt.cpu().numpy()

    def finalize(self, *args):
        pass


class WindSpeed(Diagnostics):
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
        self.diagnostic.channels = ["u10m", "v10m"]

    def _init_variables(
        self,
    ):
        dims = self.get_dimensions()
        dtypes = self.get_dtype()
        var = self.subgroup.createVariable(
            "wind_speed10m", dtypes[self.diagnostic.type], dims[self.diagnostic.type]
        )
        var.setncattr('units', units["wind_speed10m"])
        var.setncattr('long_name', long_names["wind_speed10m"])

    def _init_subgroup(
        self,
    ):
        self.subgroup = self.group

    def get_dimensions(self):
        return {"wind_speed10m": ("ensemble", "time") + self.domain_dims}

    def get_dtype(self):
        return {"wind_speed10m": np.float32}

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        u10m = output[:, 0]
        v10m = output[:, 1]
        wind_speed10m = torch.sqrt(u10m**2 + v10m**2)
        self.subgroup["wind_speed10m"][
            batch_id : batch_id + batch_size, time_index
        ] = wind_speed10m.cpu().numpy()

    def finalize(self, *args):
        pass

class HeatIndex(Diagnostics):
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
        self.diagnostic.channels = ["t2m", "d2m"]
        lookup_table = xarray.open_zarr("/pscratch/sd/a/amahesh/hens/heat_index_lookup.zarr")
        self.lookup_table = torch.tensor(lookup_table['heat_index'].values, device=device)
        self.t2m_keys = torch.tensor(lookup_table['Ta'].values, device=device)
        self.rh2m_keys = torch.tensor(lookup_table['Rh'].values, device=device)

    def _init_variables(
        self,
    ):
        dims = self.get_dimensions()
        dtypes = self.get_dtype()
        var = self.subgroup.createVariable(
            "heat_index", dtypes[self.diagnostic.type], dims[self.diagnostic.type]
        )
        var.setncattr('units', units["heat_index"])
        var.setncattr('long_name', long_names["heat_index"])

    def _init_subgroup(
        self,
    ):
        self.subgroup = self.group

    def get_dimensions(self):
        return {"heat_index": ("ensemble", "time") + self.domain_dims}

    def get_dtype(self):
        return {"heat_index": np.float32}

    def _saturation_vapor_pressure(self, temperature):
        """
        temperature must be in units of Kelvin
        """
        sat_pressure_0c = 6.112
        return sat_pressure_0c * torch.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))

    def _calculate_rh_from_dewpoint(self, t2m, d2m):
        """
        t2m: temperature at 2m in Kelvin
        d2m: dewpoint at 2m in Kelvin
        """
        sat_vapor_pressure = self._saturation_vapor_pressure(t2m)
        vapor_pressure = self._saturation_vapor_pressure(d2m)
        return vapor_pressure / sat_vapor_pressure

    def update(
        self, output: torch.Tensor, time_index: int, batch_id: int, batch_size: int
    ):
        t2m = output[:, 0].unsqueeze(-1)
        d2m = output[:, 1].unsqueeze(-1)
        rh2m = self._calculate_rh_from_dewpoint(t2m, d2m)

        t2m_diff = torch.abs(self.t2m_keys - t2m)
        rh_diff = torch.abs(self.rh2m_keys - rh2m)
        t2m_diff[self.t2m_keys > t2m] = float("inf")
        rh_diff[self.rh2m_keys > rh2m] = float("inf")
        t2m_idx = torch.argmin(t2m_diff, dim=-1)
        rh_idx = torch.argmin(rh_diff, dim=-1)

        heat_index = self.lookup_table[t2m_idx, rh_idx]

        self.subgroup["heat_index"][
            batch_id : batch_id + batch_size, time_index
        ] = heat_index.cpu().numpy()

    def finalize(self, *args):
        pass

DiagnosticTypes = {
    "raw": Raw,
    "ivt" : IVT,
    "wind_speed10m" : WindSpeed,
    "heat_index" : HeatIndex
}
