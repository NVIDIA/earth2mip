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
import torch
from typing import Literal
from earth2mip.schema import Grid
from earth2mip.diagnostic.base import DiagnosticBase, DiagnosticConfigBase


class Identity(DiagnosticBase):
    """Idenity function. You probably don't need to use this unless you know what you
    are doing. Primarly used by the factory.
    """

    def __init__(self, in_channels: str, grid: Grid):
        super().__init__()
        self.grid = grid
        self.channels = in_channels

    @property
    def in_channels(self) -> list[str]:
        return self.channels

    @property
    def out_channels(self) -> list[str]:
        return self.channels

    @property
    def in_grid(self) -> Grid:
        return self.grid

    @property
    def out_grid(self) -> Grid:
        return self.grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @classmethod
    def load_diagnostic(cls, in_channels: list[str], grid: Grid):
        return cls(in_channels, grid)

    @classmethod
    def load_config_type(cls):
        return IdentityConfig


class IdentityConfig(DiagnosticConfigBase):

    type: Literal["Identity"] = "Identity"
    in_channels: list[str]
    grid: Grid = Grid.grid_721x1440

    def initialize(self):
        return Identity.load_diagnostic(self.in_channels, self.grid)
