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

"""
FCN v2 Small adapter
"""
from typing import List
import logging
import os
import datetime
import torch
import json

import numpy as np
import onnxruntime as ort
import dataclasses

from earth2mip import registry, schema, networks, config, initial_conditions, geometry

from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet as SFNO
from modulus.utils.sfno.zenith_angle import cos_zenith_angle


class _CosZenWrapper(torch.nn.Module):
    """Wrapper for SFNO model with Cosine Zenith inputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        lon: np.ndarray,
        lat: np.ndarray,
        device,
        local_center_path: str = "",
        local_std_path: str = "",
    ):
        super().__init__()
        self.module = model
        self.lon = lon
        self.lat = lat

        # self.center = torch.as_tensor(
        #     np.load(local_center_path),
        #     device=device,
        #     dtype=torch.float32,
        # )

        # self.std = torch.as_tensor(
        #     np.load(local_std_path),
        #     device=device,
        #     dtype=torch.float32,
        # )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.center.to(x.dtype).to(x.device)) / self.std.to(x.dtype).to(
            x.device
        )

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.dtype).to(x.device) + self.center.to(x.dtype).to(
            x.device
        )

    def forward(self, x: torch.Tensor, time: datetime.datetime):
        """Cosine zenith forward model call."""
        x = x.squeeze(1)
        n_batch = x.shape[0]
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        cosz = cos_zenith_angle(time, lon_grid, lat_grid)
        cosz = cosz.astype(np.float32)
        z = torch.from_numpy(cosz).to(device=x.device)
        z = z.repeat(n_batch, *[1 for i in range(x.ndim - 1)])
        z = torch.cat([x, z], dim=1)
        x = self.forward_batch(z, x)
        x = x.unsqueeze(1)
        return x

    @torch.no_grad()
    def forward_batch(self, z: torch.Tensor, x: torch.Tensor):
        n = z.shape[0]
        for batch in range(0, n):
            # self.z.copy_(z[batch : batch + 1])
            # self.g.replay()
            # x[batch : batch + 1].copy_(self.x, non_blocking=True)
            print(z[batch : batch + 1].shape)
            x[batch : batch + 1] = self.module(z[batch : batch + 1])
        return x


def load(package, *, pretrained=True, device="cuda"):
    assert pretrained

    local_center = np.load(package.get("global_means.npy"))
    local_std = np.load(package.get("global_stds.npy"))

    with open(package.get("metadata.json")) as f:
        metadata = json.load(f)

    with open(package.get("config.json")) as json_file:
        config = json.load(json_file)

    lat = metadata["lat"]
    lon = metadata["lon"]
    model = SFNO({}, **config).to(device)
    sfno_wrapper = _CosZenWrapper(model, lon, lat, device)
    grid = schema.Grid.grid_721x1440
    channel_set = schema.ChannelSet.var73
    dt = datetime.timedelta(hours=12)

    inference = networks.Inference(
        sfno_wrapper,
        channels=None,
        center=local_center,
        scale=local_std,
        grid=grid,
        channel_names=channel_set.list_channels(),
        channel_set=channel_set,
        time_step=dt,
    )
    inference.to(device)
    return inference
