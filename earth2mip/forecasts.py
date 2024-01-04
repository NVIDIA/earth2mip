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
import asyncio
import datetime
import logging
from typing import Any, Iterator, List, Protocol, Sequence

import torch
import xarray

import earth2mip.grid
import earth2mip.initial_conditions
from earth2mip import time_loop

logger = logging.getLogger(__name__)


class Forecast(Protocol):
    @property
    def channel_names(self) -> List[str]:
        pass

    @property
    def grid(self) -> earth2mip.grid.LatLonGrid:
        pass

    def __getitem__(self, i: int) -> Iterator[torch.Tensor]:
        """Shape of returned tensor is (1, n_channels, n_lat, n_lon)"""
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


class select_channels(Forecast):
    def __init__(self, forecast: Forecast, channel_names: list[str]):
        self.forecast = forecast
        self._channel_names = channel_names
        self._index = [self.forecast.channel_names.index(x) for x in self.channel_names]

    @property
    def channel_names(self):
        return self._channel_names

    @property
    def grid(self):
        return self.forecast.grid

    async def __getitem__(self, i: int):
        async for x in self.forecast[i]:
            yield x[:, self._index]


class TimeLoopForecast(Forecast):
    """Wrap an fcn-mip TimeLoop object as a forecast"""

    def __init__(
        self,
        time_loop: time_loop.TimeLoop,
        times: Sequence[datetime.datetime],
        data_source: earth2mip.initial_conditions.base.DataSource,
    ):
        self._data_source = data_source
        self.time_loop = time_loop
        self._times = times

    @property
    def channel_names(self):
        return self.time_loop.out_channel_names

    @property
    def grid(self):
        return self.time_loop.grid

    async def __getitem__(self, i):
        x = earth2mip.initial_conditions.get_initial_condition_for_model(
            self.time_loop, self._data_source, time=self._times[i]
        )
        count = 0
        dt = self._times[1] - self._times[0]
        yield_every = int(dt // self.time_loop.time_step)
        assert yield_every * self.time_loop.time_step == dt  # noqa
        for time, data, _ in self.time_loop(x=x, time=self._times[i]):
            if count % yield_every == 0:
                logger.info("forecast %s", time)
                yield data
            count += 1


class XarrayForecast(Forecast):
    """Turn an xarray into a forecast-like dataset"""

    def __init__(
        self,
        ds: xarray.Dataset,
        fields,
        times: Sequence[datetime.datetime],
        device,
    ):
        self._dataset = ds
        self._fields = fields
        self._times = times
        self.device = device

    @property
    def channel_names(self):
        return self._fields

    @property
    def grid(self):
        lat = self._dataset.lat.values.tolist()
        lon = self._dataset.lon.values.tolist()
        return earth2mip.grid.LatLonGrid(lat, lon)

    async def __getitem__(self, i) -> torch.Tensor:
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
            yield torch.from_numpy(data.values).to(self.device)[None]
