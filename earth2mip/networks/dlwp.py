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

import datetime
import logging

import modulus
import numpy as np
import torch
import xarray
from modulus.utils.filesystem import Package
from modulus.utils.zenith_angle import cos_zenith_angle

import earth2mip.grid
from earth2mip import networks

logger = logging.getLogger(__file__)

CHANNELS = ["t850", "z1000", "z700", "z500", "z300", "tcwv", "t2m"]

# TODO: Added here explicitly for better access. This will be imported from:
# modulus repo after this PR is merged: https://github.com/NVIDIA/modulus/pull/138
class _DLWPWrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        lsm,
        longrid,
        latgrid,
        topographic_height,
        ll_to_cs_mapfile_path,
        cs_to_ll_mapfile_path,
    ):
        super(_DLWPWrapper, self).__init__()
        self.model = model
        self.lsm = lsm
        self.longrid = longrid
        self.latgrid = latgrid
        self.topographic_height = topographic_height

        # load map weights
        self.input_map_wts = xarray.open_dataset(ll_to_cs_mapfile_path)
        self.output_map_wts = xarray.open_dataset(cs_to_ll_mapfile_path)

    @property
    def channel_names():
        return CHANNELS

    def prepare_input(self, input, time):
        device = input.device
        dtype = input.dtype

        i = self.input_map_wts.row.values - 1
        j = self.input_map_wts.col.values - 1
        data = self.input_map_wts.S.values
        M = torch.sparse_coo_tensor(np.array((i, j)), data).type(dtype).to(device)

        bs, t, chan = input.shape[0], input.shape[1], input.shape[2]
        input = input.reshape(bs * t * chan, -1) @ M.T
        input = input.reshape(bs, t, chan, 6, 64, 64)
        input_list = list(torch.split(input, 1, dim=1))
        input_list = [tensor.squeeze(1) for tensor in input_list]
        repeat_vals = (input.shape[0], -1, -1, -1, -1)  # repeat along batch dimension
        for i in range(len(input_list)):
            tisr = np.maximum(
                cos_zenith_angle(
                    time
                    - datetime.timedelta(hours=6 * (input.shape[0] - 1))
                    + datetime.timedelta(hours=6 * i),
                    self.longrid,
                    self.latgrid,
                ),
                0,
            ) - (
                1 / np.pi
            )  # subtract mean value
            tisr = (
                torch.tensor(tisr, dtype=dtype)
                .to(device)
                .unsqueeze(dim=0)
                .unsqueeze(dim=0)
            )  # add channel and batch size dimension
            tisr = tisr.expand(*repeat_vals)  # TODO - find better way to batch TISR
            input_list[i] = torch.cat(
                (input_list[i], tisr), dim=1
            )  # concat along channel dim

        input_model = torch.cat(
            input_list, dim=1
        )  # concat the time dimension into channels

        lsm_tensor = torch.tensor(self.lsm, dtype=dtype).to(device).unsqueeze(dim=0)
        lsm_tensor = lsm_tensor.expand(*repeat_vals)
        topographic_height_tensor = (
            torch.tensor((self.topographic_height - 3.724e03) / 8.349e03, dtype=dtype)
            .to(device)
            .unsqueeze(dim=0)
        )
        topographic_height_tensor = topographic_height_tensor.expand(*repeat_vals)

        input_model = torch.cat(
            (input_model, lsm_tensor, topographic_height_tensor), dim=1
        )
        return input_model

    def prepare_output(self, output):
        device = output.device
        dtype = output.dtype
        output = torch.split(output, output.shape[1] // 2, dim=1)
        output = torch.stack(output, dim=1)  # add time dimension back in
        i = self.output_map_wts.row.values - 1
        j = self.output_map_wts.col.values - 1
        data = self.output_map_wts.S.values
        M = torch.sparse_coo_tensor(np.array((i, j)), data).type(dtype).to(device)

        output = output.reshape(output.shape[0], 2, output.shape[2], -1) @ M.T
        output = output.reshape(output.shape[0], 2, output.shape[2], 721, 1440)

        return output

    def forward(self, x, time):
        x = self.prepare_input(x, time)
        y = self.model(x)
        return self.prepare_output(y)


def load(package: Package, *, pretrained=True, device="cuda"):
    assert pretrained  # noqa

    # load static datasets
    lsm = xarray.open_dataset(package.get("land_sea_mask_rs_cs.nc"))["lsm"].values
    topographic_height = xarray.open_dataset(package.get("geopotential_rs_cs.nc"))[
        "z"
    ].values
    latlon_grids = xarray.open_dataset(package.get("latlon_grid_field_rs_cs.nc"))
    latgrid, longrid = latlon_grids["latgrid"].values, latlon_grids["longrid"].values

    # load maps
    ll_to_cs_mapfile_path = package.get("map_LL721x1440_CS64.nc")
    cs_to_ll_mapfile_path = package.get("map_CS64_LL721x1440.nc")

    with torch.cuda.device(device):
        core_model = modulus.Module.from_checkpoint(package.get("dlwp.mdlus"))
        model = _DLWPWrapper(
            core_model,
            lsm,
            longrid,
            latgrid,
            topographic_height,
            ll_to_cs_mapfile_path,
            cs_to_ll_mapfile_path,
        )

        channel_names = ["t850", "z1000", "z700", "z500", "z300", "tcwv", "t2m"]
        center = np.load(package.get("global_means.npy"))
        scale = np.load(package.get("global_stds.npy"))
        grid = earth2mip.grid.equiangular_lat_lon_grid(721, 1440)
        dt = datetime.timedelta(hours=12)
        inference = networks.Inference(
            model,
            center=center,
            scale=scale,
            grid=grid,
            channel_names=channel_names,
            time_step=dt,
            n_history=1,
        )
        inference.to(device)
        return inference
