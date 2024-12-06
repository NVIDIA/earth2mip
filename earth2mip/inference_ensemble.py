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

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Optional

import cftime
import numpy as np
import torch
import tqdm
import xarray
from modulus.distributed.manager import DistributedManager
from netCDF4 import Dataset as DS

import earth2mip.grid

__all__ = ["run_inference", "run_basic_inference"]

# need to import initial conditions first to avoid unfortunate
# GLIBC version conflict when importing xarray. There are some unfortunate
# issues with the environment.
from earth2mip import initial_conditions, regrid, time_loop
from earth2mip._channel_stds import channel_stds
from earth2mip.ensemble_utils import (
    generate_bred_vector,
    generate_noise_correlated,
    generate_noise_grf,
)
from earth2mip.netcdf import initialize_netcdf, update_netcdf
from earth2mip.networks import get_model
from earth2mip.schema import EnsembleRun, PerturbationStrategy
from earth2mip.time_loop import TimeLoop

logger = logging.getLogger("inference")


def get_checkpoint_path(rank, batch_id, path):
    directory = os.path.join(path, f"{rank}")
    filename = f"{batch_id}.pth"
    return os.path.join(directory, filename)


def save_restart(restart, rank, batch_id, path):
    path = get_checkpoint_path(rank, batch_id, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Saving restart file to {path}.")
    torch.save(restart, path)


def run_ensembles(
    *,
    n_steps: int,
    weather_event,
    model: TimeLoop,
    perturb,
    x,
    nc,
    domains,
    n_ensemble: int,
    batch_size: int,
    rank: int,
    output_frequency: int,
    output_grid: Optional[earth2mip.grid.LatLonGrid],
    date_obj: datetime,
    restart_frequency: Optional[int],
    output_path: str,
    restart_initial_directory: str = "",
    progress: bool = True,
):
    if not output_grid:
        output_grid = model.grid

    regridder = regrid.get_regridder(model.grid, output_grid).to(model.device)

    diagnostics = initialize_netcdf(nc, domains, output_grid, n_ensemble, model.device)
    initial_time = date_obj
    time_units = initial_time.strftime("hours since %Y-%m-%d %H:%M:%S")
    nc["time"].units = time_units
    nc["time"].calendar = "standard"

    for batch_id in range(0, n_ensemble, batch_size):
        logger.info(f"ensemble members {batch_id+1}-{batch_id+batch_size}/{n_ensemble}")
        batch_size = min(batch_size, n_ensemble - batch_id)

        x = x.repeat(batch_size, 1, 1, 1, 1)
        x_start = perturb(x, rank, batch_id, model.device)
        # restart_dir = weather_event.properties.restart

        # TODO: figure out if needed
        # if restart_dir:
        #     path = get_checkpoint_path(rank, batch_id, restart_dir)
        #     # TODO use logger
        #     logger.info(f"Loading from restart from {path}")
        #     kwargs = torch.load(path)
        # else:
        #     kwargs = dict(
        #         x=x,
        #         normalize=False,
        #         time=time,
        #     )

        iterator = model(initial_time, x_start)

        # Check if stdout is connected to a terminal
        if sys.stderr.isatty() and progress:
            iterator = tqdm.tqdm(iterator, total=n_steps)

        time_count = -1

        # for time, data, restart in iterator:

        for k, (time, data, _) in enumerate(iterator):
            # if restart_frequency and k % restart_frequency == 0:
            #     save_restart(
            #         restart,
            #         rank,
            #         batch_id,
            #         path=os.path.join(output_path, "restart", time.isoformat()),
            #     )

            # Saving the output
            if output_frequency and k % output_frequency == 0:
                time_count += 1
                logger.debug(f"Saving data at step {k} of {n_steps}.")
                nc["time"][time_count] = cftime.date2num(time, nc["time"].units)
                update_netcdf(
                    regridder(data),
                    diagnostics,
                    domains,
                    batch_id,
                    time_count,
                    model.grid,
                    model.out_channel_names,
                )

            if k == n_steps:
                break

        # if restart_frequency is not None:
        #     save_restart(
        #         restart,
        #         rank,
        #         batch_id,
        #         path=os.path.join(output_path, "restart", "end"),
        #     )


def main(config=None):
    logging.basicConfig(level=logging.INFO)

    if config is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("config")
        parser.add_argument("--weather_model", default=None)
        args = parser.parse_args()
        config = args.config

    # If config is a file
    if os.path.exists(config):
        config: EnsembleRun = EnsembleRun.parse_file(config)
    # If string, assume JSON string
    elif isinstance(config, str):
        config: EnsembleRun = EnsembleRun.model_validate(json.loads(config))
    # Otherwise assume parsable obj
    else:
        raise ValueError(
            f"Passed config parameter {config} should be valid file or JSON string"
        )

    # if args and args.weather_model:
    #     config.weather_model = args.weather_model

    # Set up parallel
    DistributedManager.initialize()
    device = DistributedManager().device
    group = torch.distributed.group.WORLD

    logging.info(f"Earth-2 MIP config loaded {config}")
    logging.info(f"Loading model onto device {device}")
    model = get_model(config.weather_model, device=device)
    logging.info("Constructing initializer data source")
    perturb = get_initializer(
        model,
        config,
    )
    logging.info("Running inference")
    run_inference(model, config, perturb, group)


def get_initializer(
    model,
    config,
):
    def perturb(x, rank, batch_id, device):
        shape = x.shape
        if config.perturbation_strategy == PerturbationStrategy.gaussian:
            noise = config.noise_amplitude * torch.normal(
                torch.zeros(shape), torch.ones(shape)
            ).to(device)
        elif config.perturbation_strategy == PerturbationStrategy.correlated:
            noise = generate_noise_correlated(
                shape,
                reddening=config.noise_reddening,
                device=device,
                noise_amplitude=config.noise_amplitude,
            )
        elif config.perturbation_strategy == PerturbationStrategy.spherical_grf:
            noise = generate_noise_grf(
                shape,
                model.grid,
                sigma=config.grf_noise_sigma,
                alpha=config.grf_noise_alpha,
                tau=config.grf_noise_tau,
                device=device,
            )
        elif config.perturbation_strategy == PerturbationStrategy.bred_vector:
            noise = generate_bred_vector(
                x,
                model,
                config.noise_amplitude,
                time=config.weather_event.properties.start_time,
            )
        elif config.perturbation_strategy == PerturbationStrategy.none:
            return x
        if rank == 0 and batch_id == 0:  # first ens-member is deterministic
            noise[0, :, :, :, :] = 0

        # When field is not in known normalization dictionary set scale to 0
        scale = []
        for i, channel in enumerate(model.in_channel_names):
            if channel in channel_stds:
                scale.append(channel_stds[channel])
            else:
                scale.append(0)
        scale = torch.tensor(scale, device=x.device)

        if config.perturbation_channels is None:
            return x + noise * scale[:, None, None]
        else:
            channel_list = model.in_channel_names
            indices = torch.tensor(
                [
                    channel_list.index(channel)
                    for channel in config.perturbation_channels
                    if channel in channel_list
                ]
            )
            x[:, :, indices, :, :] += (
                noise[:, :, indices, :, :] * scale[indices, None, None]
            )
        return x

    return perturb


def run_basic_inference(
    model: time_loop.TimeLoop,
    n: int,
    data_source: Any,
    time: datetime,
):
    """Run a basic inference"""

    x = initial_conditions.get_initial_condition_for_model(model, data_source, time)

    arrays = []
    times = []
    for k, (time, data, _) in enumerate(model(time, x)):
        arrays.append(data.cpu().numpy())
        times.append(time)
        if k == n:
            break

    stacked = np.stack(arrays)
    coords = dict(lat=model.grid.lat, lon=model.grid.lon)
    coords["channel"] = model.out_channel_names
    coords["time"] = times
    return xarray.DataArray(
        stacked, dims=["time", "history", "channel", "lat", "lon"], coords=coords
    )


def run_inference(
    model: TimeLoop,
    config: EnsembleRun,
    perturb: Any = None,
    group: Any = None,
    progress: bool = True,
    # TODO add type hints
    data_source: Any = None,
):
    """Run an ensemble inference for a given config and a perturb function

    Args:
        group: the torch distributed group to use for the calculation
        progress: if True use tqdm to show a progress bar
        data_source: a Mapping object indexed by datetime and returning an
            xarray.Dataset object.
    """
    if not perturb:
        perturb = get_initializer(model, config)

    if not group and torch.distributed.is_initialized():
        group = torch.distributed.group.WORLD

    weather_event = config.get_weather_event()

    if not data_source:
        data_source = initial_conditions.get_data_source(
            model.in_channel_names,
            initial_condition_source=weather_event.properties.initial_condition_source,
            netcdf=weather_event.properties.netcdf,
        )

    date_obj = weather_event.properties.start_time
    x = initial_conditions.get_initial_condition_for_model(model, data_source, date_obj)

    dist = DistributedManager()
    n_ensemble_global = config.ensemble_members
    n_ensemble = n_ensemble_global // dist.world_size
    if n_ensemble == 0:
        logger.warning("World size is larger than global number of ensembles.")
        n_ensemble = n_ensemble_global

    # Set random seed
    seed = config.seed
    torch.manual_seed(seed + dist.rank)
    np.random.seed(seed + dist.rank)

    if config.output_dir:
        date_str = "{:%Y_%m_%d_%H_%M_%S}".format(date_obj)
        name = weather_event.properties.name
        output_path = (
            f"{config.output_dir}/"
            f"Output.{config.weather_model}."
            f"{name}.{date_str}"
        )
    else:
        output_path = config.output_path

    if not os.path.exists(output_path):
        # Avoid race condition across ranks
        os.makedirs(output_path, exist_ok=True)

    if dist.rank == 0:
        # Only rank 0 copies config files over
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            f.write(config.model_dump_json())

    group_rank = torch.distributed.get_group_rank(group, dist.rank)
    output_file_path = os.path.join(output_path, f"ensemble_out_{group_rank}.nc")

    with DS(output_file_path, "w", format="NETCDF4") as nc:
        # assign global attributes
        nc.model = config.weather_model
        nc.config = config.model_dump_json()
        nc.weather_event = weather_event.model_dump_json()
        nc.date_created = datetime.now().isoformat()
        nc.history = " ".join(sys.argv)
        nc.institution = "NVIDIA"
        nc.Conventions = "CF-1.10"

        run_ensembles(
            weather_event=weather_event,
            model=model,
            perturb=perturb,
            nc=nc,
            domains=weather_event.domains,
            x=x,
            n_ensemble=n_ensemble,
            n_steps=config.simulation_length,
            output_frequency=config.output_frequency,
            batch_size=config.ensemble_batch_size,
            rank=dist.rank,
            date_obj=date_obj,
            restart_frequency=config.restart_frequency,
            output_path=output_path,
            output_grid=(
                earth2mip.grid.from_enum(config.output_grid)
                if config.output_grid
                else None
            ),
            progress=progress,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier(group)

    logger.info(f"Ensemble forecast finished, saved to: {output_file_path}")


if __name__ == "__main__":
    main()
