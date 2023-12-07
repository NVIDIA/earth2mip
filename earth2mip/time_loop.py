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

import dataclasses
import datetime
from typing import Any, Iterator, List, Optional, Protocol, Tuple, TypeVar

import pandas as pd
import torch

import earth2mip.grid

ChannelNameT = str


@dataclasses.dataclass
class GeoTensorInfo:
    """Metadata explaining how tensor maps onto the Earth

    Describes a tensor ``x`` with shape ``(batch, history, channel, lat, lon)``.

    """

    channel_names: List[ChannelNameT]
    grid: earth2mip.grid.LatLonGrid
    n_history_levels: int = 1
    history_time_step: datetime.timedelta = datetime.timedelta(hours=0)


class TimeLoop(Protocol):
    """Abstract protocol that a custom time loop must follow

    This is a callable which yields time and output information. Some attributes
    are required to define the input and output data required.

    The expectation is that this class and the data passed to it are on the same
    device. While torch modules can be moved between devices easily, this is not
    true for all frameworks.

    Attributes:
        in_channel_names:
        out_channel_names:
        grid:
        n_history_levels:
        history_time_step:
        time_step:
        device:

    """

    # TODO refactor TimeLoop to take in GeoTensorInfo
    in_channel_names: List[ChannelNameT]
    out_channel_names: List[ChannelNameT]
    grid: earth2mip.grid.LatLonGrid
    n_history_levels: int = 1
    history_time_step: datetime.timedelta = datetime.timedelta(hours=0)
    time_step: datetime.timedelta
    device: torch.device
    dtype: torch.dtype = torch.float32

    def __call__(
        self, time: datetime.datetime, x: torch.Tensor, restart: Optional[Any] = None
    ) -> Iterator[Tuple[datetime.datetime, torch.Tensor, Any]]:
        """
        Args:
            x: an initial condition. has shape (B, n_history_levels,
                len(in_channel_names), Y, X).  (Y, X) should be consistent with
                ``grid``. The history dimension is in increasing order, so the
                current state corresponds to x[:, -1].  Specifically, ``x[:,
                -i]`` is the data correspond to ``time - (i-1) *
                self.history_time_step``.
            time: the datetime to start with, by default assumed to be in UTC.
            restart: if provided this restart information (typically some torch
                Tensor) can be used to restart the time loop

        Yields:
            (time, output, restart) tuples. ``output`` is a tensor with
                shape (B, len(out_channel_names), Y, X) which will be used for
                diagnostics. Restart data should encode the state of the time
                loop.
        """
        pass


StateT = TypeVar("StateT")


class TimeStepper(Protocol[StateT]):
    """An functional interface that can be used for time stepping

        state -> (state, output)

    This uses a generic state, but concrete Tensors as input and output.  This
    allows users to directly control the time-stepping logic and potentially
    modify the state in model-specific manner, but the basic initial conditions
    and running outputs are concrete torch Tensors.

    One example is the graphcast time stepper. Graphcast uses jax and xarray to
    handle the state.

    It should be used like this::

        stepper = MyStepper()
        state = stepper.initialize(x, time)

        outputs = []
        for i in range(10):
            state, output = stepper.step(state)
            outputs.append(output)

    One benefit is that the state can be saved and reloaded trivially to restart
    the simulation.

    """

    @property
    def input_info(self) -> GeoTensorInfo:
        pass

    @property
    def output_info(self, state: StateT) -> GeoTensorInfo:
        pass

    @property
    def device(self, state: StateT) -> torch.device:
        pass

    @property
    def dtype(self, state: StateT) -> torch.device:
        pass

    @property
    def time_step(self) -> datetime.timedelta:
        pass

    def initialize(self, x: torch.Tensor, time: datetime.datetime) -> StateT:
        """

        x is described by ``self.input_info``
        """
        pass

    def step(self, state: StateT) -> tuple[StateT, torch.Tensor]:
        """step the state and return the ml output as a tensor

        The output tensor is described by ``self.output_info``
        """


class TimeStepperLoop(TimeLoop):
    """Turn a TimeStepper into a TimeLoop"""

    def __init__(self, stepper: TimeStepper):
        self.stepper = stepper

    @property
    def in_channel_names(self) -> List[ChannelNameT]:
        return self.stepper.input_info.channel_names

    @property
    def out_channel_names(self) -> List[ChannelNameT]:
        return self.stepper.output_info.channel_names

    @property
    def grid(self) -> earth2mip.grid.LatLonGrid:
        return self.stepper.input_info.grid

    @property
    def n_history_levels(self) -> int:
        return self.stepper.input_info.n_history_levels

    @property
    def history_time_step(self) -> datetime.timedelta:
        return self.stepper.input_info.history_time_step

    @property
    def time_step(self) -> datetime.timedelta:
        return self.stepper.time_step

    @property
    def device(self) -> torch.device:
        return self.stepper.device

    @property
    def dtype(self) -> torch.dtype:
        return self.stepper.dtype

    def __call__(
        self, time: datetime.datetime, x: torch.Tensor, restart: Optional[Any] = None
    ) -> Iterator[Tuple[datetime.datetime, torch.Tensor, Any]]:
        if restart is None:
            state = self.stepper.initialize(x, time)
        else:
            state = restart

        # get initial targets with nan filled for channels not in input
        output = torch.full(
            [
                x.shape[0],
                len(self.stepper.output_info.channel_names),
                *self.stepper.output_info.grid.shape,
            ],
            dtype=x.dtype,
            fill_value=torch.nan,
            device=x.device,
        )
        index = pd.Index(self.in_channel_names)
        indexer = index.get_indexer(self.out_channel_names)
        mask = indexer != -1
        output[:, mask] = x[:, -1, indexer[mask]]
        yield time, output, state

        while True:
            state, output = self.stepper.step(state)
            time += self.time_step
            assert output.ndim == 4  # noqa
            yield time, output, state
