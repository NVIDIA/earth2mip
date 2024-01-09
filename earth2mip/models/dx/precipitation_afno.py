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
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch

from earth2mip import config, grid
from earth2mip.geo_operator import GeoOperator
from earth2mip.model_registry import ModelRegistry, Package
from earth2mip.models.nn.afno_precip import PrecipNet

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


class PrecipitationBase(torch.nn.Module):
    def __init__(
        self,
        core_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
    ):
        super().__init__()
        self.core_model = core_model
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.eps = 1e-5

    input_coords = OrderedDict(
        {
            "variable": np.array(IN_CHANNELS),
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    output_coords = OrderedDict(
        {
            "variable": np.array(["tp"]),
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    @classmethod
    def load_default_package(cls) -> Package:
        # TODO
        return Package(
            "ngc://model/nvidia/modulus/modulus_diagnostics@v0.1/precipitation_afno.zip"
        )

    @classmethod
    def load_model(cls, package: Package) -> GeoOperator:

        model = PrecipNet.from_checkpoint(package.get("precipitation_afno.mdlus"))
        model.eval()

        input_center = torch.Tensor(np.load(package.get("global_means.npy")))
        input_scale = torch.Tensor(np.load(package.get("global_stds.npy")))

        return cls(model, input_center, input_scale)

    # TODO: Change to __call__
    def forward_step(
        self,
        x: torch.Tensor,
        coords: OrderedDict[str, np.ndarray],
    ) -> Tuple[torch.Tensor, OrderedDict[str, np.ndarray]]:
        # TODO: Add handshake checks
        x = (x - self.center) / self.scale
        out = self.core_model(x)
        # Unlog output
        # https://github.com/NVlabs/FourCastNet/blob/master/utils/weighted_acc_rmse.py#L66
        out = self.eps * (torch.exp(out) - 1)

        output_coords = coords.copy()
        output_coords["variable"] = self.output_coords["variable"]
        return out, output_coords  # Softmax channels


class PrecipitationAFNO(PrecipitationBase, GeoOperator):
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
        super().__init__(model, in_center, in_scale)

    @property
    def in_channel_names(self) -> list[str]:
        return self.input_coords["variable"]

    @property
    def out_channel_names(self) -> list[str]:
        return self.output_coords["variable"]

    @property
    def in_grid(self) -> grid.LatLonGrid:
        return grid.equiangular_lat_lon_grid(720, 1440, includes_south_pole=False)

    @property
    def out_grid(self) -> grid.LatLonGrid:
        return grid.equiangular_lat_lon_grid(720, 1440, includes_south_pole=False)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4  # noqa
        input_coords = self.input_coords.copy()
        input_coords["batch"] = np.arange(0, x.shape[0])
        out, _ = PrecipitationBase.forward_step(self, x, input_coords)
        return out

    @classmethod
    def load_package(
        cls, registry: str = os.path.join(config.MODEL_REGISTRY, "diagnostics")
    ) -> Package:
        model_registry = ModelRegistry(registry)
        return model_registry.get_model("e2mip://precipitation_afno")

    @classmethod
    def load_diagnostic(cls, package: Package, device: str = "cuda:0") -> GeoOperator:

        model = PrecipNet.from_checkpoint(package.get("precipitation_afno.mdlus"))
        model.eval()

        input_center = torch.Tensor(np.load(package.get("global_means.npy")))
        input_scale = torch.Tensor(np.load(package.get("global_stds.npy")))

        return cls(model, input_center, input_scale).to(device)
