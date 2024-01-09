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
from earth2mip.models.diagnostic.base import DiagnosticModel
from earth2mip.models.nn.cgnet import CGNetModule


class ClimateNetBase(torch.nn.Module):
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

    input_coords = OrderedDict(
        {
            "variable": np.array(["tcwv", "u850", "v850", "msl"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    output_coords = OrderedDict(
        {
            "variable": np.array(["climnet_bg", "climnet_tc", "climnet_ar"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(90, -90, 1440, endpoint=False),
        }
    )

    @classmethod
    def load_default_package(cls) -> Package:
        # TODO
        return Package(
            "ngc://model/nvidia/modulus/modulus_diagnostics@v0.1/climatenet.zip"
        )

    @classmethod
    def load_model(cls, package: Package) -> DiagnosticModel:

        model = CGNetModule(
            channels=cls.input_coords["variable"],
            classes=cls.output_coords["variable"],
        )
        weights_path = package.get("weights.tar")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()

        center = torch.Tensor(np.load(package.get("global_means.npy")))[:, None, None]
        scale = torch.Tensor(np.load(package.get("global_stds.npy")))[:, None, None]

        return cls(model, center, scale)

    # TODO: Change to __call__
    def forward_step(
        self,
        x: torch.Tensor,
        coords: OrderedDict[str, np.ndarray],
    ) -> Tuple[torch.Tensor, OrderedDict[str, np.ndarray]]:
        # TODO: Add handshake checks
        x = (x - self.center) / self.scale
        out = self.core_model(x)

        output_coords = coords.copy()
        output_coords["variable"] = self.output_coords["variable"]
        return torch.softmax(out, 1), output_coords  # Softmax channels


class ClimateNet(ClimateNetBase, GeoOperator):
    """Climate Net Diagnostic model, built into Earth-2 MIP. This model can be used to
    create prediction labels for tropical cyclones and atmopheric rivers. Produces
    non-standard output channels climnet_bg, climnet_tc and climnet_ar representing
    background label, tropical cyclone and atmopheric river labels.

    Note:
        This model and checkpoint are from Prabhat et al. 2021
        https://doi.org/10.5194/gmd-14-107-2021
        https://github.com/andregraubner/ClimateNet

    Example:
        >>> package = ClimateNet.load_package()
        >>> model = ClimateNet.load_diagnostic(package)
        >>> x = torch.randn(1, 4, 721, 1440)
        >>> out = model(x)
        >>> out.shape
        (1, 3, 721, 1440)
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
        return grid.equiangular_lat_lon_grid(721, 1440)

    @property
    def out_grid(self) -> grid.LatLonGrid:
        return grid.equiangular_lat_lon_grid(721, 1440)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4  # noqa
        input_coords = self.input_coords.copy()
        input_coords["batch"] = np.arange(0, x.shape[0])
        out, _ = ClimateNetBase.forward_step(self, x, input_coords)
        return out

    @classmethod
    def load_package(
        cls, registry: str = os.path.join(config.MODEL_REGISTRY, "diagnostics")
    ) -> Package:
        model_registry = ModelRegistry(registry)
        return model_registry.get_model("e2mip://climatenet")

    @classmethod
    def load_diagnostic(cls, package: Package, device: str = "cuda:0") -> GeoOperator:

        model = CGNetModule(
            channels=cls.input_coords["variable"].shape[0],
            classes=cls.output_coords["variable"].shape[0],
        )
        weights_path = package.get("weights.tar")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        center = torch.Tensor(np.load(package.get("global_means.npy")))[:, None, None]
        scale = torch.Tensor(np.load(package.get("global_stds.npy")))[:, None, None]

        return cls(model, center, scale).to(device)
