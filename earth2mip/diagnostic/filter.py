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
from typing import Literal, Optional
from modulus.distributed import DistributedManager
from earth2mip.schema import Grid
from earth2mip.model_registry import Package
from earth2mip.diagnostic.base import DiagnosticBase, DiagnosticConfigBase


class Filter(DiagnosticBase):
    """Simply filters channels of a given input array. Probably shouldn't be using this
    manually unless you know what you are doing.

    Note
    ----
    If this errors its likely because you have a mismatch of channels

    Example
    -------
    >>> dfilter = Filter(['u10m','t2m'], ['u10m'], Grid.grid_721x1440)
    >>> x = torch.randn(1, 2, 721, 1440)
    >>> out = dfilter(x)
    >>> out.shape
    (1, 1, 721, 1440)
    """

    def __init__(self, in_channels: list[str], out_channels: list[str], grid: Grid):
        super().__init__()
        self.grid = grid

        self._in_channels = in_channels
        self._out_channels = out_channels

        indexes_list = []
        try:
            for channel in self._out_channels:
                indexes_list.append(self._in_channels.index(channel))
            self.register_buffer("indexes", torch.IntTensor(indexes_list))
        except ValueError as e:
            raise ValueError(
                "Looks like theres a mismatch between input and "
                + f"requested channels. {e}"
            )

    @property
    def in_channels(self) -> list[str]:
        return self._in_channels

    @property
    def out_channels(self) -> list[str]:
        return self._out_channels

    @property
    def in_grid(self) -> Grid:
        return self.grid

    @property
    def out_grid(self) -> Grid:
        return self.grid

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        print(x.device, self.indexes.device)
        return torch.index_select(x, 1, self.indexes)

    @classmethod
    def load_diagnostic(
        cls,
        package: Optional[Package],
        in_channels: list[str],
        out_channels: list[str],
        grid: Grid,
        device: str = "cuda:0",
    ):
        return cls(in_channels, out_channels, grid).to(device)

    @classmethod
    def load_config_type(cls):
        return FilterConfig


class FilterConfig(DiagnosticConfigBase):

    type: Literal["Filter"] = "Filter"
    in_channels: list[str]
    out_channels: list[str]
    grid: Grid = Grid.grid_721x1440

    def initialize(self):
        dm = DistributedManager()
        package = Filter.load_package()
        return Filter.load_diagnostic(
            package, self.in_channels, self.out_channels, self.grid, device=dm.device
        )