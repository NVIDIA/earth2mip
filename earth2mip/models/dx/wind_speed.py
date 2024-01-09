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
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch

from earth2mip import grid
from earth2mip.geo_operator import GeoOperator


class WindSpeedBase(torch.nn.Module):
    def __init__(self, level: int):
        self.level = level

    @property
    def input_coords(self) -> OrderedDict[str, np.ndarray]:
        return OrderedDict({"variable": np.array([f"u{self.level}", f"v{self.level}"])})

    @property
    def output_coords(self) -> OrderedDict[str, np.ndarray]:
        return OrderedDict({"variable": np.array([f"ws{self.level}"])})

    # TODO: Change to __call__
    def forward_step(
        self,
        x: torch.Tensor,
        coords: OrderedDict[str, np.ndarray],
    ) -> Tuple[torch.Tensor, OrderedDict[str, np.ndarray]]:
        # TODO: Add handshake checks
        output_coords = coords.copy()
        output_coords["variable"] = self.output_coords["variable"]
        return torch.sqrt(x[:, 0:1, ...] ** 2 + x[:, 1:2, ...] ** 2), output_coords


class WindSpeed(WindSpeedBase, GeoOperator):
    """Computes the wind speed at a given level.
    This is largely just an example of what a diagnostic calculation could look like.

    Example:
        >>> windspeed = WindSpeed('10m', Grid.grid_721x1440)
        >>> x = torch.randn(1, 2, 721, 1440)
        >>> out = windspeed(x)
        >>> out.shape
        (1, 1, 721, 1440)
    """

    def __init__(self, level: int, grid: grid.LatLonGrid):
        super().__init__(level)
        self.grid = grid

    @property
    def in_channel_names(self) -> list[str]:
        return self.input_coords["variable"]

    @property
    def out_channel_names(self) -> list[str]:
        return self.output_coords["variable"]

    @property
    def in_grid(self) -> grid.LatLonGrid:
        return self.grid

    @property
    def out_grid(self) -> grid.LatLonGrid:
        return self.grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        out, _ = WindSpeedBase.forward_step(self, x, OrderedDict({}))
        return out
