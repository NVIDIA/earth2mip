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

import torch.distributed
import typer
from distributed import Client
from modulus.distributed.manager import DistributedManager
from functools import partial
from earth2mip.ensemble_utils import CorrelatedSphericalField

from earth2mip import inference_ensemble, networks, score_ensemble_outputs
from earth2mip.schema import EnsembleRun
import torch_harmonics as th

logging.basicConfig(
    format="%(asctime)s:%(levelname)-s:%(name)s:%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__file__)
WD = os.getcwd()


def get_distributed_client(rank, shard):
    scheduler_file = "/pscratch/sd/a/amahesh/scheduler_{:04d}_2.json".format(shard)
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
    assert shard < n_shards  # noqa

    time = datetime.datetime(1, 1, 1)

    config_path = os.path.join(root, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    DistributedManager.initialize()
    dist = DistributedManager()
    #model = networks.get_model(config["model"], device=dist.device)

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
    client = get_distributed_client(dist.rank, shard)
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
        
        if config["model"] == 'multicheckpoint':                       
            model_names = [
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed26",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed27",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed28",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed29",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed30",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed31",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed70",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed71",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed72",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed74",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed76",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed12",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed16",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed17",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed18",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed77",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed78",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed80",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed81",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed84",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed86",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed87",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed90",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed91",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed92",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed93",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed94",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed96",
                "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed98",
            ]
            for model_idx, model_name in enumerate(model_names):            
                model = networks.get_model(model_name, device=dist.device)                
                #sampler = CorrelatedSphericalField(720, 750 * 1000, 6.0, 0.2, channel_names=model.channel_names).to(model.device)
                #if dist.rank != 0:
                #    model.source = sampler 
                logging.info("Constructing initializer data source")        
                perturb = inference_ensemble.get_initializer(                                  
                    model,                                                  
                    run,                                                 
                )                                                           
                logging.info("Running inference")                           
                run.weather_event.properties.start_time = initial_time
                run.output_path = d
                inference_ensemble.run_inference(model, run, perturb=perturb, group=group, model_idx=model_idx, num_models=len(model_names))
        else:
            #model = networks.get_model(config["model"], device=dist.device)
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
