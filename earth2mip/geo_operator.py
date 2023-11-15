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
from abc import abstractmethod
from typing import Protocol

import torch

from earth2mip import grid


class GeoOperator(Protocol):
    """Geo Operator

    This is the most primative functional of Earth-2 MIP which represents a
    operators on geospatial fields. This implies the following two requirements:
        1) The operation must define in and out channel variables representing the
            fields in the input/output arrays.
        2) The operation must define the in and out grid schemas.

    Many auto-gressive models can be represented as a GeoOperator and can maintain a
    internal state. Diagnostic models must be a GeoOperator by definition.

    Warning:
        Geo Function is a concept not full adopted in Earth-2 MIP and is being adopted
        progressively.
    """

    @property
    def in_channel_names(self) -> list[str]:
        pass

    @property
    def out_channel_names(self) -> list[str]:
        pass

    @property
    def in_grid(self) -> grid.LatLonGrid:
        pass

    @property
    def out_grid(self) -> grid.LatLonGrid:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of geographic data of shape
            [..., in_chans, lat, lon]

        Returns:
            torch.Tensor: Output tensor of geographic data of shape
            [..., out_chans, lat, lon]
        """
        pass
