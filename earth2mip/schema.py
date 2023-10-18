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

from typing import List, Optional, Mapping, Any
import pydantic
from earth2mip import weather_events
from earth2mip.weather_events import InitialConditionSource, WeatherEvent
from enum import Enum
import datetime
import numpy as np

__all__ = ["InitialConditionSource", "WeatherEvent"]


class Grid(Enum):
    grid_721x1440 = "721x1440"
    grid_720x1440 = "720x1440"
    s2s_challenge = "s2s"

    @property
    def shape(self):
        if self == Grid.grid_721x1440:
            return (721, 1440)
        elif self == Grid.grid_720x1440:
            return (720, 1440)
        elif self == Grid.s2s_challenge:
            return (181, 360)
        else:
            raise ValueError(f"Unknown grid {self}")

    @property
    def lat(self):
        return _grids[self]["lat"]

    @property
    def lon(self):
        return _grids[self]["lon"]


_grids = {
    Grid.grid_721x1440: {
        "lat": np.linspace(90, -90.0, 721),
        "lon": np.linspace(0, 359.75, 1440),
    },
    Grid.grid_720x1440: {
        "lat": np.linspace(89.75, -90.0, 720),
        "lon": np.linspace(0, 359.75, 1440),
    },
    Grid.s2s_challenge: {
        "lat": np.linspace(90, -90.0, 181),
        "lon": np.linspace(0, 359, 360),
    },
}


class InferenceEntrypoint(pydantic.BaseModel):
    """
    Attrs:
        name: an entrypoint string like ``my_package:model_entrypoint``.
            this points to a function ``model_entrypoint(package)`` which returns an
            ``Inference`` object given a package
        kwargs: the arguments to pass to the constructor
    """

    name: str = ""
    kwargs: Mapping[Any, Any] = pydantic.Field(default_factory=dict)


class Model(pydantic.BaseModel):
    """Metadata for using a ERA5 time-stepper model

    Attrs:
        entrypoint: if provided, will be used to load a custom time-loop
            implementation.

    """

    n_history: int = 0
    grid: Grid = Grid.grid_720x1440
    in_channels_names: List[str] = pydantic.Field(default_factory=list)
    out_channels_names: List[str] = pydantic.Field(default_factory=list)
    architecture: str = ""
    architecture_entrypoint: str = ""
    time_step: datetime.timedelta = datetime.timedelta(hours=6)
    entrypoint: Optional[InferenceEntrypoint] = None


class PerturbationStrategy(str, Enum):
    correlated = "correlated"
    gaussian = "gaussian"
    bred_vector = "bred_vector"
    spherical_grf = "spherical_grf"
    none = "none"
