# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

from typing import Optional, Any
import argparse
import logging
import os
import sys
from datetime import datetime
import xarray
import cftime

import numpy as np
import torch
import tqdm
from modulus.distributed.manager import DistributedManager
from netCDF4 import Dataset as DS

__all__ = ["run_inference"]

# need to import initial conditions first to avoid unfortunate
# GLIBC version conflict when importing xarray. There are some unfortunate
# issues with the environment.
from earth2mip import initial_conditions, time_loop
from earth2mip.ensemble_utils import (
    generate_noise_correlated,
    draw_noise,
    generate_noise_grf,
    generate_correlated_spherical_grf,
    load_spherical_mean_covar,
    generate_bred_vector,
)
from earth2mip.netcdf import finalize_netcdf, initialize_netcdf, update_netcdf
from earth2mip.networks import get_model, Inference
from earth2mip.schema import EnsembleRun, Grid, PerturbationStrategy
from earth2mip.time import convert_to_datetime
from earth2mip import regrid


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
    model,
    perturb,
    nc,
    domains,
    ds,
    n_ensemble: int,
    batch_size: int,
    device: str,
    rank: int,
    output_frequency: int,
    output_grid: Optional[Grid],
    date_obj: datetime,
    restart_frequency: Optional[int],
    output_path: str,
    restart_initial_directory: str = "",
    progress: bool = True,
):

    if not output_grid:
        output_grid = model.grid

    regridder = regrid.get_regridder(model.grid, output_grid).to(device)

    # TODO infer this from the model
    ds = ds.astype(np.float32)
    assert not np.any(np.isnan(ds))

    if output_grid == model.grid:
        lat = ds.lat.values
        lon = ds.lon.values
    else:
        lat, lon = regridder.lat, regridder.lon

    diagnostics = initialize_netcdf(
        nc, domains, output_grid, lat, lon, n_ensemble, device
    )
    time = convert_to_datetime(ds.time[-1])
    time_units = time.strftime("hours since %Y-%m-%d %H:%M:%S")
    nc["time"].units = time_units
    nc["time"].calendar = "standard"

    for batch_id in range(0, n_ensemble, batch_size):
        logger.info(f"ensemble members {batch_id+1}-{batch_id+batch_size}/{n_ensemble}")
        time = convert_to_datetime(ds.time[-1])
        batch_size = min(batch_size, n_ensemble - batch_id)

        x = torch.from_numpy(ds.values)[None].to(device)
        x = model.normalize(x)
        x = x.repeat(batch_size, 1, 1, 1, 1)
        perturb(x, rank, batch_id, device)
        restart_dir = weather_event.properties.restart
        if restart_dir:
            path = get_checkpoint_path(rank, batch_id, restart_dir)
            # TODO use logger
            logger.info(f"Loading from restart from {path}")
            kwargs = torch.load(path)
        else:
            kwargs = dict(
                x=x,
                normalize=False,
                time=time,
            )

        iterator = model.run_steps_with_restart(n=n_steps, **kwargs)

        # Check if stdout is connected to a terminal
        if sys.stderr.isatty() and progress:
            iterator = tqdm.tqdm(iterator, total=n_steps)

        k = 0
        time_count = -1

        for time, data, restart in iterator:
            k += 1

            if restart_frequency and k % restart_frequency == 0:
                save_restart(
                    restart,
                    rank,
                    batch_id,
                    path=os.path.join(output_path, "restart", time.isoformat()),
                )

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
                    model,
                    lat,
                    lon,
                    ds.channel,
                )

        if restart_frequency is not None:
            save_restart(
                restart,
                rank,
                batch_id,
                path=os.path.join(output_path, "restart", "end"),
            )

    finalize_netcdf(diagnostics, nc, domains, weather_event, model.channel_set)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--fcn_model", default=None)
    args = parser.parse_args()

    config: EnsembleRun = EnsembleRun.parse_file(args.config)
    if args.fcn_model:
        config.fcn_model = args.fcn_model

    # Set up parallel
    DistributedManager.initialize()
    group = torch.distributed.group.WORLD
    model = get_model(config.fcn_model)
    perturb = get_initializer(
        model,
        config,
    )
    run_inference(model, config, perturb, group)


def get_initializer(
    model,
    config,
):
    if config.perturbation_strategy == PerturbationStrategy.correlated_spherical_grf:
        mean, sqrt_covar = load_spherical_mean_covar()

    def perturb(x, rank, batch_id, device):
        shape = x.shape
        if config.perturbation_strategy == PerturbationStrategy.gaussian:
            noise = config.noise_amplitude * torch.normal(
                torch.zeros(shape), torch.ones(shape)
            ).to(device)
        elif config.perturbation_strategy == PerturbationStrategy.spherical_grf:
            noise = generate_noise_grf(
                shape,
                model.grid,
                sigma=config.grf_noise_sigma,
                alpha=config.grf_noise_alpha,
                tau=config.grf_noise_tau,
            ).to(device)
        elif (
            config.perturbation_strategy
            == PerturbationStrategy.correlated_spherical_grf
        ):
            noise = generate_correlated_spherical_grf(
                shape,
                model.grid,
                sigma=config.grf_noise_sigma,
                alpha=config.grf_noise_alpha,
                tau=config.grf_noise_tau,
                mean=mean,
                sqrt_covar=sqrt_covar,
            ).to(device)
        elif config.perturbation_strategy == PerturbationStrategy.correlated:
            noise = generate_noise_correlated(
                shape,
                reddening=config.noise_reddening,
                device=device,
                noise_amplitude=config.noise_amplitude,
            )
        elif config.perturbation_strategy == PerturbationStrategy.gp:
            corr = torch.load("/lustre/fsw/sw_climate_fno/ensemble_init_stats/corr.pth")
            length_scales = torch.load(
                "/lustre/fsw/sw_climate_fno/ensemble_init_stats/length_scales.pth"
            ).to(device)
            noise = config.noise_amplitude * draw_noise(
                corr, spreads=None, length_scales=length_scales, device=device
            ).to(device)
            noise = noise.repeat(config.ensemble_batch_size, 1, 1, 1, 1)
        elif config.perturbation_strategy == PerturbationStrategy.bred_vector:
            noise = generate_bred_vector(
                x,
                model,
                config.noise_amplitude,
                time=config.weather_event.properties.start_time,
            )
        if rank == 0 and batch_id == 0:  # first ens-member is deterministic
            noise[0, :, :, :, :] = 0
        if config.single_value_perturbation:
            x[:, :, 5, :, :] += noise[:, :, 5, :, :]
        else:
            x += noise
        return x

    return perturb


def run_basic_inference(model: time_loop.TimeLoop, n: int, data_source, time):
    """Run a basic inference"""
    ds = data_source[time].sel(channel=model.in_channel_names)
    x = torch.from_numpy(ds.values).cuda()
    # need a batch dimension of length 1
    x = x[None]

    arrays = []
    times = []
    for k, (time, data, _) in enumerate(model(time, x)):
        arrays.append(data.cpu().numpy())
        times.append(time)
        if k == n:
            break

    stacked = np.stack(arrays)
    coords = {**ds.coords}
    coords["channel"] = model.out_channel_names
    coords["time"] = times
    return xarray.DataArray(
        stacked, dims=["time", "history", "channel", "lat", "lon"], coords=coords
    )


def run_inference(
    model: Inference,
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
            model.n_history,
            model.grid,
            model.channel_set,
            initial_condition_source=weather_event.properties.initial_condition_source,
            netcdf=weather_event.properties.netcdf,
        )

    ds = data_source[weather_event.properties.start_time]

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

    date_obj = convert_to_datetime(ds.time[-1])

    if config.output_dir:
        date_str = "{:%Y_%m_%d_%H_%M_%S}".format(date_obj)[5:]
        name = weather_event.properties.name
        output_path = f"{config.output_dir}/Output.{config.fcn_model}.{name}.{date_str}"
    else:
        output_path = config.output_path

    if not os.path.exists(output_path):
        # Avoid race condition across ranks
        os.makedirs(output_path, exist_ok=True)

    if dist.rank == 0:
        # Only rank 0 copies config files over
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            f.write(config.json())

    model.to(dist.device)
    group_rank = torch.distributed.get_group_rank(group, dist.rank)
    output_file_path = os.path.join(output_path, f"ensemble_out_{group_rank}.nc")

    with DS(output_file_path, "w", format="NETCDF4") as nc:
        # assign global attributes
        nc.model = config.fcn_model
        nc.config = config.json()
        nc.weather_event = weather_event.json()
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
            ds=ds,
            n_ensemble=n_ensemble,
            n_steps=config.simulation_length,
            output_frequency=config.output_frequency,
            batch_size=config.ensemble_batch_size,
            rank=dist.rank,
            device=dist.device,
            date_obj=date_obj,
            restart_frequency=config.restart_frequency,
            output_path=output_path,
            output_grid=config.output_grid,
            progress=progress,
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier(group)

    logger.info(f"Ensemble forecast finished, saved to: {output_file_path}")


if __name__ == "__main__":
    main()
