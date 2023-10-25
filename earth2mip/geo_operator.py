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
from typing import Protocol
from abc import abstractmethod
from earth2mip.schema import Grid


class GeoOperator(Protocol):
    """Geo Operator

    This is the most primative functional of Earth-2 MIP which represents a
    operators on geographical data to produce geographical data. This implies the
    following two requirements:
        1) The operation must define in and out channel variables representing the
            fields in the input/output arrays.
        2) The operation must define the in and out grid schemas.

    Many auto-gressive models can be represented as a GeoOperator and can maintain a
    internal state. Diagnostic models must be a GeoOperator by definition.

    Warning
    -------
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
    def in_grid(self) -> Grid:
        pass

    @property
    def out_grid(self) -> Grid:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass