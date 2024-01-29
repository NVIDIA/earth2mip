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

import datetime
from enum import Enum
from typing import Any, List, Mapping, Optional

import pydantic

from earth2mip import weather_events
from earth2mip.weather_events import InitialConditionSource, WeatherEvent

__all__ = [
    "InitialConditionSource",
    "WeatherEvent",
    "EnsembleRun",
    "InferenceEntrypoint",
    "PerturbationStrategy",
]


class Grid(Enum):
    grid_721x1440 = "721x1440"
    grid_720x1440 = "720x1440"
    s2s_challenge = "s2s"


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


class PerturbationStrategy(Enum):
    correlated = "correlated"
    gaussian = "gaussian"
    bred_vector = "bred_vector"
    spherical_grf = "spherical_grf"
    none = "none"


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
        perturbation_channels: channel(s) perturbed by the initial condition perturbation strategy, None = all channels
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
    perturbation_channels: Optional[List[str]] = None
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
