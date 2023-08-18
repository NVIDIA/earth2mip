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

from typing import Protocol, List, Iterator, Tuple, Any, Optional
import datetime
import torch
from earth2mip.schema import Grid


ChannelNameT = str


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

    in_channel_names: List[ChannelNameT]
    out_channel_names: List[ChannelNameT]
    grid: Grid
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
                ``grid``. ``x[:, -i]`` is the data correspond to
                ``time - (i-1) * self.history_time_step``. Note this means that
                ``time`` corresponds to ``x[:, -1]``...not ``x[:, 0]``.
            time: the datetime to start with
            restart: if provided this restart information (typically some torch
                Tensor) can be used to restart the time loop

        Yields:
            (time, output, restart) tuples. ``output`` is a tensor with
                shape (B, len(out_channel_names), Y, X) which will be used for
                diagnostics. Restart data should encode the state of the time
                loop.
        """
        pass
