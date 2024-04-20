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

from collections import deque

import torch
import torch.distributed


async def yield_lagged_ensembles(
    *,
    observations,
    forecast,
    lags: int = 2,
    n: int = 10,
):
    """Yield centered lagged ensembles

    The forecast array has shape (len(observations), n)

    The ensemble consist of runs initialized with an offset of (-lags, ..., 0,
    ...lags). The ensemble size is therefore ``2*lags + =`` for points within
    the interior of the array.

    Supports running in parallel using the ``rank`` and ``world_size`` flags
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1

    # example one. no garbage collection
    nt = len(observations)
    assert n < nt  # noqa

    # work trackers that will be used to determine when an ensemble is finished,
    # and ensure that all data is processed
    finished = set()
    ensemble = {}

    obs_buffer = deque([])

    for i in range(n + world_size):
        obs_buffer.append(await observations[i])

    n_iter = int(nt // world_size)
    assert nt % world_size == 0, "{}".format(nt)  # noqa

    buffers = None

    for i0 in range(n_iter):
        for k in range(world_size):
            i = world_size * i0 + k
            if i + n + 1 < nt:
                obs_buffer.append(await observations[i + n + 1])

        i = world_size * i0 + rank
        nsteps = min(nt - world_size * i0 - 1, n)

        lead_time = -1
        async for y in forecast[i]:
            lead_time += 1
            j = i + lead_time
            if lead_time > nsteps:
                break

            if torch.distributed.is_initialized():
                buffers = [torch.empty_like(y) for _ in range(world_size)]
                # TODO only gather from needed ranks (i - m)
                torch.distributed.all_gather(buffers, y)
                if y.device != torch.device("cpu"):
                    cpu_buffers = [
                        torch.empty_like(b, device="cpu", pin_memory=True)
                        for b in buffers
                    ]
                    for cpu, gpu in zip(cpu_buffers, buffers):
                        cpu.copy_(gpu, non_blocking=True)
                else:
                    cpu_buffers = buffers
            else:
                cpu_buffers = [y]

            lead_time = j - i
            # need to loop over ranks to ensure that number of iterations
            # per rank is the same
            for r in range(world_size):
                for m in range(-lags, lags + 1):
                    ii = i0 * world_size + r
                    jj = ii + lead_time

                    if jj >= nt:
                        break

                    # Should this rank process the data or not?
                    i_owner = jj - lead_time - m
                    if i_owner % world_size != rank:
                        continue

                    k = (jj, lead_time + m)

                    store_me = cpu_buffers[r]
                    # ensemble[k][m]
                    ensemble.setdefault(k, {})[m] = store_me
                    # There are two options for this finishing criteria
                    # 1. if it work is not done in the next iteration, then we know
                    # we are done this would be implemented by
                    #
                    #       if not done(j, lead_time + m, i + 1):
                    #
                    # 2. if the ensemble has the expected number of members
                    # 2 seems easier to parallelize and less subject to the
                    # looping we take, so is what we do here:
                    expected = num(n=n, ell=lead_time + m, j=jj, L=lags)
                    if jj < nt and len(ensemble[k]) == expected:
                        # sanity check that a single ensemble is not
                        # processed multiple times
                        if k in finished:
                            assert False, k  # noqa
                        finished.add(k)
                        # need to synchronize to ensure cpu buffers are filled
                        # before yielding the complete ensemble
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        yield k, ensemble.pop(k), await observations[jj]

        for _ in range(world_size):
            obs_buffer.popleft()

    assert not ensemble, len(ensemble)  # noqa


def num(n, ell, j, L):
    a = max(ell - j, ell - n, -L)
    b = min(ell, L)
    return b - a + 1


def done(j, ell, i, lags, n):
    """Unused helper function wich says if lag ell and valid_time j are written
    to in a given iteration `i` of the loop in lagged_average_simple

    This is one way to implement the done criteria which is less easily
    parallelized. I am leaving it in the code for educational value only.
    """
    #
    a = j - i - lags <= ell <= j - i + lags
    b = n >= j - i >= 0
    return a & b
