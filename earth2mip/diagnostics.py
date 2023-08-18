# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

import torch
import numpy as np
from datetime import datetime
from earth2mip.weather_events import CWBDomain, Window, MultiPoint
from earth2mip import weather_events
from earth2mip.schema import Grid
from netCDF4._netCDF4 import Group
from typing import Union


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

    def finalize(
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

    def finalize(self, *args):
        pass


DiagnosticTypes = {
    "raw": Raw,
}


def extended_best_track_reader(storm, t_init, t_finit):
    stormname = storm[:-4]
    stormyear = storm[-4:]
    ExtendedBestTrack = "./ExtendedBestTrack_combined.txt"
    vmax = []
    mslp = []
    rmax = []
    lat = []
    lon = []
    time = []
    with open(ExtendedBestTrack, "r") as EBT:
        for line in EBT:
            if stormname.upper() in line[9:20] and stormyear in line[28:32]:
                time_ebt = line[28:32] + line[21:27] + "00:00"
                date_obj = datetime.strptime(time_ebt, "%Y%m%d%H%M:%S")
                day_of_year = date_obj.timetuple().tm_yday - 1
                hour_of_day = date_obj.timetuple().tm_hour
                mytime = 24 * day_of_year + hour_of_day - t_init
                if mytime < 0.0 or mytime > t_finit:
                    continue
                time.append(mytime)
                vmax.append(float(line[46:49]))
                rmax.append(float(line[55:58]))
                lat.append(float(line[34:38]))
                lon.append(float(line[41:45]))
                mslp.append(float(line[50:54]))

    vmax = np.array(vmax)
    vmax[vmax < 0.0] = np.nan
    rmax = np.array(rmax)
    rmax[rmax < 0.0] = np.nan
    lat = np.array(lat)
    lat[np.where(lat < 0.0)] = np.nan
    lon = np.array(lon)
    lon = np.subtract(360.0, lon)
    lon[np.where(lon < 0.0)] = np.nan
    mslp = np.array(mslp)
    mslp[mslp < 0.0] = np.nan
    return time, lat, lon, mslp, vmax, rmax


def tropical_cyclone_tracker(z_850, z_250, u_850, v_850):
    vorticity = compute_vorticity(u_850, v_850)
    dZ = np.subtract(z_250, z_850)
    i_z850, j_z850 = np.unravel_index(z_850.argmin(), z_850.shape)
    i_dz, j_dz = np.unravel_index(dZ.argmax(), dZ.shape)
    i_v, j_v = np.unravel_index(vorticity.argmax(), vorticity.shape)
    max_tilt_i = np.square(np.max([i_dz, i_v, i_z850]) - np.min([i_dz, i_v, i_z850]))
    max_tilt_j = np.square(np.max([j_dz, j_v, j_z850]) - np.min([j_dz, j_v, j_z850]))
    max_tilt = np.sqrt(max_tilt_i + max_tilt_j)
    if max_tilt > 6:  # tilt larger than 150km and its not a TC
        i_z850 = np.nan
        j_z850 = np.nan
    return i_z850, j_z850


def compute_vorticity(u, v, dx=25000.0, dy=25000.0):
    dudx = np.gradient(u, axis=0) / dx
    dvdy = np.gradient(v, axis=1) / dy
    vorticity = np.add(dvdy, dudx)
    return vorticity


def exceedance_probability(V, V_thresh=33.0):
    V = np.array(V)
    V_thresh = np.array(V_thresh)
    exceedance_prob = np.zeros(V.shape)
    exceedance_prob[V > V_thresh] = 1
    return exceedance_prob


def emanuel_damage_function(V, V_thresh=25.0, V_half=77.0):
    """
    Returns the Emanuel damage function,
    Equ. 1 in Emanuel 2011: Global warming effects on U.S. hurricane damage
    """
    Vn = np.divide(
        np.clip(np.subtract(V, V_thresh), 0.0, None), np.subtract(V_half, V_thresh)
    )
    Vn3 = np.power(Vn, 3)
    return np.divide(Vn3, 1.0 + Vn3)
