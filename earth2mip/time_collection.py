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
import json
import logging
import os
import shutil
from typing import Optional

import torch.distributed
import typer
from distributed import Client
from modulus.distributed.manager import DistributedManager

from earth2mip import inference_ensemble, networks, score_ensemble_outputs
from earth2mip.initial_conditions.base import DataSource
from earth2mip.schema import EnsembleRun
from earth2mip.time_loop import TimeLoop

__all__ = ["run_over_initial_times"]

logging.basicConfig(
    format="%(asctime)s:%(levelname)-s:%(name)s:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__file__)
WD = os.getcwd()


def get_distributed_client(rank: int, n_workers: int) -> Client:
    scheduler_file = "scheduler.json"
    if rank == 0:
        client = Client(n_workers=n_workers, threads_per_worker=1)
        client.write_scheduler_file(scheduler_file)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if rank != 0:
        client = Client(scheduler_file=scheduler_file)

    return client


def main(
    root: str,
    shard: int = 0,
    n_shards: int = 1,
) -> None:
    """
    Args:
        root: the root directory of the output
        shard: index of the shard
        n_shards: split the input times into this many shards
    """

    DistributedManager.initialize()
    dist = DistributedManager()

    config_path = os.path.join(root, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    protocol = config["protocol"]

    run_over_initial_times(
        data_source=None,
        output_path=root,
        time_loop=networks.get_model(config["model"], device=dist.device),
        config=EnsembleRun.parse_obj(protocol["inference_template"]),
        initial_times=[
            datetime.datetime.fromisoformat(line.strip()) for line in protocol["times"]
        ],
        time_averaging_window=protocol.get("time_averaging_window", ""),
        score=protocol.get("score", False),
        save_ensemble=protocol.get("save_ensemble", False),
        shard=shard,
        n_shards=n_shards,
    )


def run_over_initial_times(
    *,
    time_loop: TimeLoop,
    data_source: Optional[DataSource],
    initial_times: list[datetime.datetime],
    config: EnsembleRun,
    output_path: str,
    time_averaging_window: str = "",
    score: bool = False,
    save_ensemble: bool = False,
    shard: int = 0,
    n_shards: int = 1,
    n_post_processing_workers: int = 32,
) -> None:
    """Perform a set of forecasts across many initial conditions in parallel
    with post processing

    Once complete, the data at ``output_path`` can be opened as an xarray object using
    :py:func:`earth2mip.datasets.hindcast.open_forecast`.

    Parallelizes across the available GPUs using MPI, and can be further
    parallelized across multiple MPI jobs using the ``shard``/ ``n_shards``
    flags. It can be resumed after interruption.

    Args:
        time_loop: the earth2mip TimeLoop to be evaluated. Often returned by `earth2mip.networks.get_model`
        data_source: the data source used to initialize the time_loop, overrides
            any data source specified in ``config``
        initial_times: the initial times evaluated over
        n_shards: split the input times into this many shards
        time_averaging_window: if provided, average the output over this interval. Same
            syntax as pandas.Timedelta (e.g. "2w"). Default is no time averaging.
        score: if true, score the times during the post processing
        save_ensemble: if true, then save all the ensemble members in addition to the mean
        shard: index of the shard. useful for SLURM array jobs
        n_shards: number of shards total.
        n_post_processing_workers: The number of dask distributed workers to
            devote to ensemble post processing.

    """
    assert shard < n_shards  # noqa
    assert config.weather_event
    dist = DistributedManager()
    root = output_path
    model = time_loop

    time = datetime.datetime(1, 1, 1)
    initial_times = initial_times[shard::n_shards]
    logger.info(
        f"Working on shard {shard+1}/{n_shards}. {len(initial_times)} initial times to run."
    )

    n_ensemble_batches = config.ensemble_members // config.ensemble_batch_size
    ranks_per_time = min(n_ensemble_batches, dist.world_size)
    ranks_per_time = ranks_per_time - dist.world_size % ranks_per_time

    time_rank = int(dist.rank // ranks_per_time)
    n_time_groups = int(dist.world_size // ranks_per_time)

    group_ranks = list(
        range(time_rank * ranks_per_time, (time_rank + 1) * ranks_per_time)
    )
    logger.info(
        "distributed info: " + str((dist.rank, time_rank, group_ranks, ranks_per_time))
    )

    if torch.distributed.is_initialized():
        group = torch.distributed.new_group(group_ranks)
        group_rank = torch.distributed.get_group_rank(group, dist.rank)
        initial_times = initial_times[time_rank::n_time_groups]
    else:
        group = None
        group_rank = 0

    # setup dask client for post processing
    client = get_distributed_client(dist.rank, n_workers=n_post_processing_workers)
    post_process_task = None

    # write time information to config.json:protocol.times as expected by
    # earth2mip.datasets.hindcast.open_forecast this is needed for compatibility
    # with the make_job/earth2mip.time_collection command line interface.  Do
    # not overwite the file if it already exists.
    config_path = os.path.join(output_path, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(
                {"protocol": {"times": [t.isoformat() for t in initial_times]}}, f
            )

    # begin the loop over initial times
    count = 0
    for initial_time in initial_times:
        count += 1
        start = time.now()
        output = f"{root}/{initial_time.isoformat()}"
        if os.path.exists(output):
            continue

        d = output + ".tmp"

        if torch.distributed.is_initialized():
            torch.distributed.barrier(group)

        perturb = inference_ensemble.get_initializer(
            model,
            config,
        )
        config.weather_event.properties.start_time = initial_time
        config.output_path = d
        inference_ensemble.run_inference(
            model,
            config,
            group=group,
            progress=False,
            perturb=perturb,
            data_source=data_source,
        )

        if group_rank == 0:

            def post_process(d: str) -> None:
                output_path = f"{d}/output/"
                score_ensemble_outputs.main(
                    input_path=d,
                    output_path=output_path,
                    time_averaging_window=time_averaging_window,
                    score=score,
                    save_ensemble=save_ensemble,
                )
                shutil.move(output_path, output)
                shutil.rmtree(d, ignore_errors=True)

            # do not work on more than one post processing job at once
            if post_process_task is not None:
                post_process_task.result()

            post_process_task = client.submit(post_process, d)
            stop = time.now()
            elapsed = stop - start
            remaining = elapsed * (len(initial_times) - count)
            logger.info(
                f"{count}/{len(initial_times)}: {initial_time} done. Elapsed: {elapsed.total_seconds()}s. Remaining: {remaining}s"  # noqa
            )

    # finish up final task
    if group_rank == 0 and post_process_task is not None:
        post_process_task.result()
        client.close()

    # keep barrier at end so
    # dask distributed client is not cleaned up
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    logging.info(f"rank {dist.rank} Finished.")


if __name__ == "__main__":
    typer.run(main)
