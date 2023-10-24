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
import datetime
from typing import Optional, Any, Iterator, Tuple, List
from earth2mip.diagnostic.base import DiagnosticBase
from earth2mip.time_loop import TimeLooper


class DiagnosticTimeLooper(TimeLooper):
    def __init__(
        self, diagnostics: List[DiagnosticBase], model: TimeLooper, concat: bool = True
    ):

        self.model = model
        self.diagnostics = diagnostics
        self.concat = concat

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
        if restart:
            yield from self._iterate(**restart)
        else:
            yield from self._iterate(x=x, time=time)

    def _iterate(self, x, normalize=True, time=None):
        """Yield (time, unnormalized data, restart) tuples

        restart = (time, unnormalized data)
        """

        iterator = self.model(x, time, normalize=normalize)

        for (time, data, restart) in iterator:

            out = []
            for function in self.diagnostics:
                out.append(function(data))

            out = torch.cat(out, axis=1)
            if self.concat:
                out = torch.cat([data, out], axis=1)

            yield time, out, restart
