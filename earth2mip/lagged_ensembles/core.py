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
import torch.distributed

__all__ = ["yield_lagged_ensembles"]


async def repeat(val):
    while True:
        yield val


async def yield_lagged_ensembles(
    *,
    observations,
    forecast,
    max_lag: int = 2,
    min_lag: int = -2,
    n: int = 10,
):
    """Yield lagged ensembles

    The forecast array has shape (len(observations), n)

    The ensemble consist of runs initialized with an offset of (-min_lag, ...
    ..., max_lag). The ensemble size is therefore ``2*lags + =`` for points within
    the interior of the array.

    Supports running in parallel using the ``rank`` and ``world_size`` flags.
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

    n_iter = nt // world_size
    if nt % world_size != 0:
        n_iter += 1

    recv_buffer = None

    for i0 in range(n_iter):
        i = world_size * i0 + rank
        nsteps = min(nt - world_size * i0 - 1, n)

        lead_time = -1

        single_forecast = forecast[i] if i < nt else repeat(None)
        async for y in single_forecast:
            lead_time += 1
            if lead_time > nsteps:
                break

            # initialize the buffer for receiving data
            if recv_buffer is None:
                recv_buffer = torch.empty_like(y)

            buffers = scatter(
                rank, world_size, i0, nt, y, recv_buffer, min_lag, max_lag
            )
            cpu_buffers = _convert_ensemble_to_cpu_async(buffers)

            # need to loop over ranks to ensure that number of iterations
            # per rank is the same
            for r in cpu_buffers:
                ii = i0 * world_size + r
                jj = ii + lead_time
                if jj >= nt:
                    break
                for m in range(min_lag, max_lag + 1):
                    # Should this rank process the data or not?
                    i_owner = ii - m
                    if i_owner % world_size != rank:
                        continue

                    k = (jj, lead_time + m)

                    # ensemble[k][m] = data_from(rank=r)
                    ensemble.setdefault(k, {})[m] = cpu_buffers[r]
                    # There are two options for this finishing criteria
                    # 1. if it work is not done in the next iteration, then we know
                    # we are done this would be implemented by
                    #
                    #       if not done(j, lead_time + m, i + 1):
                    #
                    # 2. if the ensemble has the expected number of members
                    # 2 seems easier to parallelize and less subject to the
                    # looping we take, so is what we do here:
                    expected = num(
                        n=n, ell=lead_time + m, j=jj, lower=min_lag, upper=max_lag
                    )
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

    assert not ensemble, len(ensemble)


def _convert_ensemble_to_cpu_async(
    buffers: dict[int, torch.Tensor]
) -> dict[int, torch.Tensor]:
    """Maybe convert gpu ensemble to cpu ensemble asynchronously

    Before using the output be sure to synchronize the cuda stream using either

        torch.cuda.syncrhonize()

    or cuda Events.

    """
    if not buffers:
        return {}

    first_buffer = next(iter(buffers.values()))
    if first_buffer != torch.device("cpu"):
        cpu_buffers = {
            r: torch.empty_like(b, device="cpu", pin_memory=True)
            for r, b in buffers.items()
        }
        for r in buffers:
            cpu = cpu_buffers[r]
            gpu = buffers[r]
            cpu.copy_(gpu, non_blocking=True)
    else:
        cpu_buffers = buffers
    return cpu_buffers


def scatter(rank, world_size, i0, nt, y, recv_buffer, lower, upper):
    """

    constraints:
        ii = i0 * world_size + src
        jj = ii + lead_time

        i = jj - lead_time - m
        i % world_size = dst:
        lower <= m <= upper

        dst = i0 * world_size + src + lead_time - lead_time - m

        dst = (src - m) % world_size
        src = (dst + m) % world_size

    """
    if torch.distributed.is_initialized():
        buffers = {}

        # too avoid redundant comms when the # of lags > world_size
        # upper - lower + 1  <= world_size
        # upper <= world_size + lower - 1
        upper = min(upper, world_size + lower - 1)
        for src in range(world_size):
            src_initial_time = i0 * world_size + src
            if src_initial_time >= nt:
                continue

            for m in range(lower, upper + 1):
                dst = (src - m) % world_size
                if rank == src == dst:
                    buffers[rank] = y
                elif rank == src:
                    torch.distributed.send(y, dst=dst)
                elif rank == dst:
                    torch.distributed.recv(recv_buffer, src=src)
                    buffers[src] = recv_buffer.clone()
    else:
        buffers = {rank: y}

    return buffers  # noqa


def num(n, ell, j, upper: int, lower: int):
    """The number of ensemble members for a given lagged ensemble

    This functions counts the lags ``m`` which satisfy the following constraints
    from the algorithm above::

        m is an integer

        # true initial time of lagged member >=0
        i = j - (ell - m) >= 0

        # number of lead times
        0 <= ell - m <= n

        # lagged window
        lower <= m <=upper

    Args:
        n: length of simulation
        ell: lead time of non-lagged member of the ensemble
        j: valid time
        upper: upper bound on lagged window
        lower: lower bound on lagged window
    """
    a = max(ell - j, ell - n, lower)
    b = min(ell, upper)
    return b - a + 1
