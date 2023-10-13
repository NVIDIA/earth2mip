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


# Enum of channels
class ChannelSet(Enum):
    """An Enum of standard sets of channels

    These correspond to the post-processed outputs in .h5 files like this:

        73var: /lustre/fsw/sw_climate_fno/test_datasets/73var-6hourly
        34var: /lustre/fsw/sw_climate_fno/34Vars

    This concept is needed to map from integer channel numbers (e.g. [0, 1, 2]
    to physical variables).

    """

    var26 = "26var"
    var34 = "34var"
    var73 = "73var"
    var_pangu = "var_pangu"

    def list_channels(self) -> List[str]:
        """List channel names corresponding to the vocabulary"""
        return _channels[self]


_channels = {
    ChannelSet.var73: [
        "u10m",
        "v10m",
        "u100m",
        "v100m",
        "t2m",
        "sp",
        "msl",
        "tcwv",
        "u50",
        "u100",
        "u150",
        "u200",
        "u250",
        "u300",
        "u400",
        "u500",
        "u600",
        "u700",
        "u850",
        "u925",
        "u1000",
        "v50",
        "v100",
        "v150",
        "v200",
        "v250",
        "v300",
        "v400",
        "v500",
        "v600",
        "v700",
        "v850",
        "v925",
        "v1000",
        "z50",
        "z100",
        "z150",
        "z200",
        "z250",
        "z300",
        "z400",
        "z500",
        "z600",
        "z700",
        "z850",
        "z925",
        "z1000",
        "t50",
        "t100",
        "t150",
        "t200",
        "t250",
        "t300",
        "t400",
        "t500",
        "t600",
        "t700",
        "t850",
        "t925",
        "t1000",
        "r50",
        "r100",
        "r150",
        "r200",
        "r250",
        "r300",
        "r400",
        "r500",
        "r600",
        "r700",
        "r850",
        "r925",
        "r1000",
    ],
    ChannelSet.var_pangu: [
        "z1000",
        "z925",
        "z850",
        "z700",
        "z600",
        "z500",
        "z400",
        "z300",
        "z250",
        "z200",
        "z150",
        "z100",
        "z50",
        "q1000",
        "q925",
        "q850",
        "q700",
        "q600",
        "q500",
        "q400",
        "q300",
        "q250",
        "q200",
        "q150",
        "q100",
        "q50",
        "t1000",
        "t925",
        "t850",
        "t700",
        "t600",
        "t500",
        "t400",
        "t300",
        "t250",
        "t200",
        "t150",
        "t100",
        "t50",
        "u1000",
        "u925",
        "u850",
        "u700",
        "u600",
        "u500",
        "u400",
        "u300",
        "u250",
        "u200",
        "u150",
        "u100",
        "u50",
        "v1000",
        "v925",
        "v850",
        "v700",
        "v600",
        "v500",
        "v400",
        "v300",
        "v250",
        "v200",
        "v150",
        "v100",
        "v50",
        "msl",
        "u10m",
        "v10m",
        "t2m",
    ],
    ChannelSet.var34: [
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
        "u100m",
        "v100m",
        "u250",
        "v250",
        "z250",
        "t250",
        "u100",
        "v100",
        "z100",
        "t100",
        "u900",
        "v900",
        "z900",
        "t900",
    ],
    ChannelSet.var26: [
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
        "u100m",
        "v100m",
        "u250",
        "v250",
        "z250",
        "t250",
    ],
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
    channel_set: ChannelSet = ChannelSet.var34
    grid: Grid = Grid.grid_720x1440
    in_channels: List[int] = pydantic.Field(default_factory=list)
    out_channels: List[int] = pydantic.Field(default_factory=list)
    architecture: str = ""
    architecture_entrypoint: str = ""
    time_step: datetime.timedelta = datetime.timedelta(hours=6)
    entrypoint: Optional[InferenceEntrypoint] = None


class PerturbationStrategy(Enum):
    correlated = "correlated"
    gaussian = "gaussian"
    bred_vector = "bred_vector"
    spherical_grf = "spherical_grf"


class EnsembleRun(pydantic.BaseModel):
    """A configuration for running an ensemble weather forecast

    Attributes:
        weather_model: The name of the fully convolutional neural network (FCN) model to use for the forecast.
        ensemble_members: The number of ensemble members to use in the forecast.
        noise_amplitude: The amplitude of the Gaussian noise to add to the initial conditions.
        noise_reddening: The noise reddening amplitude, 2.0 was the defualt set by A.G. work.
        simulation_length: The length of the simulation in timesteps.
        output_frequency: The frequency at which to write the output to file, in timesteps.
        use_cuda_graphs: Whether to use CUDA graphs to optimize the computation.
        seed: The random seed for the simulation.
        ensemble_batch_size: The batch size to use for the ensemble.
        autocast_fp16: Whether to use automatic mixed precision (AMP) with FP16 data types.
        perturbation_strategy: The strategy to use for perturbing the initial conditions.
        ic_perturbed_channels: channel(s) perturbed by the initial condition (ic) perturbation strategy, defaults to `all_channels`
        forecast_name (optional): The name of the forecast to use (alternative to `weather_event`).
        weather_event (optional): The weather event to use for the forecast (alternative to `forecast_name`).
        output_dir (optional): The directory to save the output files in (alternative to `output_path`).
        output_path (optional): The path to the output file (alternative to `output_dir`).
        restart_frequency: if provided save at end and at the specified frequency. 0 = only save at end.
        grf_noise_alpha: tuning parameter of the Gaussian random field, see ensemble_utils.generate_noise_grf for details
        grf_noise_sigma: tuning parameter of the Gaussian random field, see ensemble_utils.generate_noise_grf for details
        grf_noise_tau: tuning parameter of the Gaussian random field, see ensemble_utils.generate_noise_grf for details

    """  # noqa

    weather_model: str
    simulation_length: int
    # TODO make perturbation_strategy an Enum (see ChannelSet)
    perturbation_strategy: PerturbationStrategy = PerturbationStrategy.correlated
    ic_perturbed_channels: Optional[List[str]] = ['all_channels']
    noise_reddening: float = 2.0
    noise_amplitude: float = 0.05
    output_frequency: int = 1
    output_grid: Optional[Grid] = None
    ensemble_members: int = 1
    seed: int = 1
    ensemble_batch_size: int = 1
    # alternatives for specifiying forecast
    forecast_name: Optional[str] = None
    weather_event: Optional[weather_events.WeatherEvent] = None
    # alternative for specifying output
    output_dir: Optional[str] = None
    output_path: Optional[str] = None
    restart_frequency: Optional[int] = None
    grf_noise_alpha: float = 2.0
    grf_noise_sigma: float = 5.0
    grf_noise_tau: float = 2.0

    def get_weather_event(self) -> weather_events.WeatherEvent:
        if self.forecast_name:
            return weather_events.read(self.forecast_name)
        else:
            return self.weather_event
