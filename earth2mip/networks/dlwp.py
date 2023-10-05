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
import torch

import numpy as np
import xarray
import json
import os
import logging
import zipfile
import urllib.request
from earth2mip import registry, schema, networks, config, initial_conditions, geometry
from earth2mip.time_loop import TimeLoop
from earth2mip.schema import Grid

from modulus.models.fcn_mip_plugin import _fix_state_dict_keys
from modulus.models.dlwp import DLWP
from modulus.utils.filesystem import Package
from modulus.utils.sfno.zenith_angle import cos_zenith_angle

logger = logging.getLogger(__file__)

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
        return ["t850", "z1000", "z700", "z500", "z300", "tcwv", "t2m"]

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


def _download_default_package(package):

    model_registry = os.environ["MODEL_REGISTRY"]
    dlwp_registry = os.path.join(model_registry, "dlwp")
    if str(dlwp_registry) != str(package.root):
        logger.info("Custom package DLWP found, aborting default package")
        return

    if not os.path.isdir(package.root):
        logger.info("Downloading DLWP model checkpoint, this may take a bit")
        urllib.request.urlretrieve(
            "https://api.ngc.nvidia.com/v2/models/nvidia/modulus/"
            + "modulus_dlwp_cubesphere/versions/v0.1/files/dlwp_cubesphere.zip",
            f"{model_registry}/dlwp_cubesphere.zip",
        )
        # Unzip
        with zipfile.ZipFile(f"{model_registry}/dlwp_cubesphere.zip", "r") as zip_ref:
            zip_ref.extractall(model_registry)
        # Clean up zip
        os.remove(f"{model_registry}/dlwp_cubesphere.zip")
    else:
        logger.info("DLWP package already found, skipping download")


def load(package, *, pretrained=True, device="cuda"):
    assert pretrained
    # Download model if needed
    _download_default_package(package)

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
        # p = package.get("model.onnx")
        with open(package.get("config.json")) as json_file:
            config = json.load(json_file)
            core_model = DLWP(
                nr_input_channels=config["nr_input_channels"],
                nr_output_channels=config["nr_output_channels"],
            )

            if pretrained:
                weights_path = package.get("weights.pt")
                weights = torch.load(weights_path)
                fixed_weights = _fix_state_dict_keys(weights, add_module=False)
                core_model.load_state_dict(fixed_weights)

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
            grid = schema.Grid.grid_721x1440
            dt = datetime.timedelta(hours=12)
            inference = networks.Inference(
                model,
                channels=None,
                center=center,
                scale=scale,
                grid=grid,
                channel_names=channel_names,
                time_step=dt,
                n_history=1,
            )
            inference.to(device)
            return inference
