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

"""Forecast abstractions

A forecast is a discrete array of ``(n_initial_times, n_lead_times)``. However
because a forecast evolves forward in time, and we do not store the whole
forecast necessarily, algorithms in fcn-mip should access ``n_lead_times`` in
sequential order. This is the purpose of the abstractions here.
"""
from typing import Sequence, Any, Protocol, Iterator, List
import datetime
import torch
import xarray
import logging
import numpy
from earth2mip import time_loop

import asyncio


logger = logging.getLogger(__name__)


class Forecast(Protocol):
    channel_names: List[str]

    def __getitem__(self, i: int) -> Iterator[torch.Tensor]:
        pass


class Persistence:
    """persistence forecast. This forecast always returns the initial condition.

    Yields (channel, lat, lon)
    """

    def __init__(self, observations: Any):
        self.observations = observations

    @property
    def channel_names(self):
        x = asyncio.run(self.obserations[0])
        return x.channel.tolist()

    async def __getitem__(self, i: int):
        x = await self.observations[i]
        while True:
            yield x


class TimeLoopForecast:
    """Wrap an fcn-mip TimeLoop object as a forecast"""

    def __init__(
        self,
        time_loop: time_loop.TimeLoop,
        times: Sequence[datetime.datetime],
        observations: Any,
    ):
        assert len(times) == len(observations)
        self.observations = observations
        self.time_loop = time_loop
        self._times = times

    @property
    def channel_names(self):
        return self.time_loop.out_channel_names

    async def __getitem__(self, i):
        # TODO clean-up this interface. pick a consistent type for ``x``.
        x = await self.observations[i]
        x = x.sel(channel=self.time_loop.in_channel_names)
        x = torch.from_numpy(x.values).cuda()
        x = x[None]
        count = 0
        dt = self._times[1] - self._times[0]
        yield_every = int(dt // self.time_loop.time_step)
        assert yield_every * self.time_loop.time_step == dt
        for time, data, _ in self.time_loop(x=x, time=self._times[i]):
            if count % yield_every == 0:
                logger.info("forecast %s", time)
                yield data
            count += 1


class XarrayForecast:
    """Turn an xarray into a forecast-like dataset"""

    def __init__(
        self, ds: xarray.Dataset, fields, times: Sequence[datetime.datetime], xp=numpy
    ):
        self._dataset = ds
        self._fields = fields
        self._times = times
        self.xp = xp

    @property
    def channel_names(self):
        return self._dataset.channel.values.tolist()

    async def __getitem__(self, i):
        initial_time = self._times[i]
        j = i - 1
        all_data = (
            self._dataset.sel(initial_time=initial_time)[self._fields]
            .to_array(dim="channel")
            .sel(channel=self._fields)
            .load()
        )
        while True:
            j += 1
            time = self._times[j]
            data = all_data.sel(time=time - initial_time)
            data.data = self.xp.asarray(data.data)
            yield j, data
