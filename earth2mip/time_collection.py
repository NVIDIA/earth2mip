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
import shutil
import os

import typer
from modulus.distributed.manager import DistributedManager
import torch.distributed

from distributed import Client

from earth2mip import inference_ensemble, networks, score_ensemble_outputs
from earth2mip.schema import EnsembleRun

logging.basicConfig(
    format="%(asctime)s:%(levelname)-s:%(name)s:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__file__)
WD = os.getcwd()


def get_distributed_client(rank):
    scheduler_file = "scheduler.json"
    if rank == 0:
        client = Client(n_workers=32, threads_per_worker=1)
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
):
    """
    Args:
        root: the root directory of the output
        shard: index of the shard
        n_shards: split the input times into this many shards
    """
    assert shard < n_shards

    time = datetime.datetime(1, 1, 1)

    config_path = os.path.join(root, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    DistributedManager.initialize()
    model = networks.get_model(config["model"])
    dist = DistributedManager()

    protocol = config["protocol"]
    lines = protocol["times"][shard::n_shards]
    logger.info(
        f"Working on shard {shard+1}/{n_shards}. {len(lines)} initial times to run."
    )

    run = EnsembleRun.parse_obj(protocol["inference_template"])
    n_ensemble_batches = run.ensemble_members // run.ensemble_batch_size
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
        lines = lines[time_rank::n_time_groups]

    # setup dask client for post processing
    client = get_distributed_client(dist.rank)
    post_process_task = None

    count = 0
    for line in lines:
        count += 1
        initial_time = datetime.datetime.fromisoformat(line.strip())
        start = time.now()
        output = f"{root}/{initial_time.isoformat()}"
        if os.path.exists(output):
            continue

        d = output + ".tmp"

        if torch.distributed.is_initialized():
            torch.distributed.barrier(group)

        perturb = inference_ensemble.get_initializer(
            model,
            run,
        )
        run.weather_event.properties.start_time = initial_time
        run.output_path = d
        inference_ensemble.run_inference(
            model, run, group=group, progress=False, perturb=perturb
        )

        if group_rank == 0:

            def post_process(d):
                output_path = f"{d}/output/"
                score_ensemble_outputs.main(
                    input_path=d,
                    output_path=output_path,
                    time_averaging_window=protocol.get("time_averaging_window", ""),
                    score=protocol.get("score", False),
                    save_ensemble=protocol.get("save_ensemble", False),
                )
                shutil.move(output_path, output)
                shutil.rmtree(d, ignore_errors=True)

            # do not work on more than one post processing job at once
            if post_process_task is not None:
                post_process_task.result()

            post_process_task = client.submit(post_process, d)

            stop = time.now()
            elapsed = stop - start
            remaining = elapsed * (len(lines) - count)
            logger.info(
                f"{count}/{len(lines)}: {initial_time} done. Elapsed: {elapsed.total_seconds()}s. Remaining: {remaining}s"  # noqa
            )

    # finish up final task
    if group_rank == 0 and post_process_task is not None:
        post_process_task.result()

    # keep barrier at end so
    # dask distributed client is not cleaned up
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    logging.info(f"rank {dist.rank} Finished.")


if __name__ == "__main__":
    typer.run(main)
