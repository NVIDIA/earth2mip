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

import os

import modulus
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modulus.models.afno import AFNO

from earth2mip import config, grid
from earth2mip.diagnostic.base import DiagnosticBase
from earth2mip.model_registry import ModelRegistry, Package

IN_CHANNELS = [
    "u10m",
    "v10m",
    "t2m",
    "sp",
    "msl",
    "t850",
    "u1000",
    "v1000",
    "z1000",
    "u850",
    "v850",
    "z850",
    "u500",
    "v500",
    "z500",
    "t500",
    "z50",
    "r500",
    "r850",
    "tcwv",
]

OUT_CHANNELS = [
    "tp",  # Total precipitation
]

# =========================== Precip Model ============================
# ======================================================================


class PeriodicPad2d(nn.Module):
    def __init__(self, pad_width):
        super(PeriodicPad2d, self).__init__()
        self.pad_width = pad_width

    def forward(self, x):
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        out = F.pad(
            out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0
        )
        return out


class PrecipNet(modulus.Module):
    def __init__(
        self,
        inp_shape,
        in_channels,
        out_channels,
        patch_size=(8, 8),
        embed_dim=768,
    ):
        super().__init__()
        self.backbone = AFNO(
            inp_shape=inp_shape,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=12,
            mlp_ratio=4.0,
            drop_rate=0.0,
            num_blocks=8,
        )
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.act = nn.ReLU()
        self.eps = 1e-5

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        # Unlog output
        # https://github.com/NVlabs/FourCastNet/blob/master/utils/weighted_acc_rmse.py#L66
        x = self.eps * (torch.exp(x) - 1)
        return x


# ============================================================================
# ============================================================================


class PrecipitationAFNO(DiagnosticBase):
    """Precipitation AFNO model. Predicts the total precipation parameter which is the
    accumulated amount of liquid and frozen water (rain or snow) with units m.

    Note:
        This checkpoint is from Parthik et al. 2022.
        https://arxiv.org/abs/2202.11214
        https://github.com/NVlabs/FourCastNet

    Example:
        >>> package = PrecipAFNO.load_package()
        >>> model = PrecipAFNO.load_diagnostic(package)
        >>> x = torch.randn(1, 4, 720, 1440)
        >>> out = model(x)
        >>> out.shape
        (1, 1, 721, 1440)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
    ):
        super().__init__()
        self.grid = grid.equiangular_lat_lon_grid(720, 1440, includes_south_pole=False)

        self._in_channels = IN_CHANNELS
        self._out_channels = OUT_CHANNELS

        self.model = model
        self.register_buffer("in_center", in_center)
        self.register_buffer("in_scale", in_scale)

    @property
    def in_channel_names(self) -> list[str]:
        return self._in_channels

    @property
    def out_channel_names(self) -> list[str]:
        return self._out_channels

    @property
    def in_grid(self) -> grid.LatLonGrid:
        return self.grid

    @property
    def out_grid(self) -> grid.LatLonGrid:
        return self.grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4  # noqa
        x = (x - self.in_center) / self.in_scale
        out = self.model(x)
        return out

    @classmethod
    def load_package(
        cls, registry: str = os.path.join(config.MODEL_REGISTRY, "diagnostics")
    ) -> Package:
        registry = ModelRegistry(registry)
        return registry.get_model("e2mip://precipitation_afno")

    @classmethod
    def load_diagnostic(cls, package: Package, device="cuda:0"):

        model = PrecipNet.from_checkpoint(package.get("precipitation_afno.mdlus"))
        model.eval()

        input_center = torch.Tensor(np.load(package.get("global_means.npy")))
        input_scale = torch.Tensor(np.load(package.get("global_stds.npy")))

        return cls(model, input_center, input_scale).to(device)
