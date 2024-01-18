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

import logging

import pytest
import torch

from earth2mip.lagged_ensembles.core import yield_lagged_ensembles

c, lat, lon = 1, 10, 13


class Observations:
    def __init__(self, device, nt=20):
        self.device = device
        self.nt = nt

    async def __getitem__(self, i):
        """
        Returns (channel, lat, lon)
        """
        if i >= len(self):
            raise KeyError(i)
        return torch.tensor([i], device=self.device)

    def __len__(self):
        return self.nt


class Forecast:
    def __init__(self, device, nt):
        self.device = device
        self.nt = nt

    async def __getitem__(self, i):
        """persistence forecast

        Yields (channel, lat, lon)
        """
        if i >= self.nt:
            raise KeyError(i)

        x = torch.zeros((2,), device=self.device)
        x[0] = i

        lead_time = -1
        while True:
            lead_time += 1
            x[1] = lead_time
            yield x.clone()


@pytest.fixture(scope="session")
def dist_info():
    try:
        torch.distributed.init_process_group(init_method="env://", backend="gloo")
    except ValueError:
        logging.warn("Could not initialize torch distributed with the gloo backend.")
        return 0, 1
    else:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return rank, world_size


@pytest.mark.parametrize("nt", [16, 20])
@pytest.mark.parametrize("min_lag,max_lag", [(-2, 2), (-2, 0)])
async def test_yield_lagged_ensembles(dist_info, nt, min_lag, max_lag, regtest):
    rank, world_size = dist_info
    device = "cpu"
    forecast = Forecast(device, nt)

    niter = 0
    async for (j, k), ens, o in yield_lagged_ensembles(
        observations=Observations(device, nt),
        forecast=forecast,
        min_lag=min_lag,
        max_lag=max_lag,
    ):
        niter += 1
        i = j - k
        # assert this process is responsible for this lagged ensemble
        assert i % world_size == rank
        assert o == j
        for m in ens:
            assert min_lag <= m <= max_lag
            arr = ens[m]
            ll = arr[1]
            assert ll == k - m
            ii = arr[0]
            assert ii == i + m

    n = torch.tensor(niter)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(n)
    print(n.item(), file=regtest)
    assert niter > 0, niter
