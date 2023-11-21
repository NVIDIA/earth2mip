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
from typing import Optional

import torch

from earth2mip import grid
from earth2mip.diagnostic.base import DiagnosticBase
from earth2mip.model_registry import Package


class WindSpeed(DiagnosticBase):
    """Computes the wind speed at a given level.
    This is largely just an example of what a diagnostic calculation could look like.

    Example:
        >>> windspeed = WindSpeed('10m', Grid.grid_721x1440)
        >>> x = torch.randn(1, 2, 721, 1440)
        >>> out = windspeed(x)
        >>> out.shape
        (1, 1, 721, 1440)
    """

    def __init__(self, level: str, grid: grid.LatLonGrid):
        super().__init__()
        self.grid = grid

        self._in_channels = [f"u{level}", f"v{level}"]
        self._out_channels = [f"ws{level}"]

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
        return torch.sqrt(x[:, 0:1, ...] ** 2 + x[:, 1:2, ...] ** 2)

    @classmethod
    def load_diagnostic(
        cls, package: Optional[Package], level: str, grid: grid.LatLonGrid
    ):
        return cls(level, grid)
