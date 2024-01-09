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
from typing import Any, Iterator, List, Optional, Tuple

import torch

from earth2mip import grid
from earth2mip.models.dx.base import DiagnosticBase
from earth2mip.time_loop import TimeLoop


def filter_channels(
    input: torch.Tensor, in_channels: list[str], out_channels: list[str]
) -> torch.Tensor:
    """Utility function used for selecting a sub set of channels

    Note:
        Right now this assumes that the channels are in the thirds to last axis.

    Args:
        input (torch.Tensor): Input tensor of shape [..., channels, lat, lon]
        in_channels (list[str]): Input channel list
        out_channels (list[str]): Output channel list
    """
    indexes_list = []
    try:
        indexes_list = [in_channels.index(channel) for channel in out_channels]
    except ValueError as e:
        raise ValueError(
            "Looks like theres a mismatch between input and "
            + f"requested channels. {e}"
        )
    indexes = torch.IntTensor(indexes_list).to(input.device)
    return torch.index_select(input, -3, indexes)


class DiagnosticTimeLoop(TimeLoop):
    """Diagnostic Timeloop. This is an iterator that executes a list of diagnostic
     models on top of a model Timeloop.

    Note:
        Presently, grids must be consistent between diagnostics and the model

    Args:
        diagnostics (List[DiagnosticBase]): List of diagnostic functions to execute
        model (TimeLoop): Model inferencer iterator
        concat (bool, optional): Concatentate diagnostic outputs with model outputs.
        Defaults to True.
    """

    def __init__(
        self, diagnostics: List[DiagnosticBase], model: TimeLoop, concat: bool = True
    ):
        self.model = model
        self.diagnostics = diagnostics
        self.concat = concat

    @property
    def in_channel_names(self) -> list[str]:
        return self.model.in_channel_names

    @property
    def out_channel_names(self) -> list[str]:
        out_names = []
        for function in self.diagnostics:
            out_names.extend(function.out_channel_names)

        if self.concat:
            out_names = self.model.out_channel_names + out_names
        return out_names

    @property
    def grid(self) -> grid.LatLonGrid:
        # TODO: Need to generalize
        return self.diagnostics[0].out_grid

    @property
    def device(self) -> torch.device:
        return self.model.device

    def __call__(
        self,
        time: datetime.datetime,
        x: torch.Tensor,
        restart: Optional[Any] = None,
    ) -> Iterator[Tuple[datetime.datetime, torch.Tensor, Any]]:
        """
        Args:
            x: an initial condition. has shape (B, n_history_levels,
                len(in_channel_names), Y, X).  (Y, X) should be consistent with
                ``grid``.
            time: the datetime to start with
            restart: if provided this restart information (typically some torch
                Tensor) can be used to restart the time loop

        Yields:
            (time, output, restart) tuples. ``output`` is a tensor with
                shape (B, len(out_channel_names), Y, X) which will be used for
                diagnostics. Restart data should encode the state of the time
                loop.
        """
        iterator = self.model(time, x, restart=restart)
        for (time, data, restart) in iterator:
            out = []
            for function in self.diagnostics:
                data0 = filter_channels(
                    data, self.model.out_channel_names, function.in_channel_names
                )
                out.append(function(data0))

            out = torch.cat(out, axis=1)
            if self.concat:
                out = torch.cat([data, out], axis=1)

            yield time, out, restart
