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

# %%
import argparse
import logging
import os
import json
from functools import partial
import torch
from modulus.distributed.manager import DistributedManager
from earth2mip.inference_ensemble import run_inference, get_initializer
from earth2mip.ensemble_utils import brown_noise
from earth2mip.networks import get_model
from earth2mip.schema import EnsembleRun


def apply_gaussian_perturbation(
     x,
     time_step,
     in_channel_names,
     device,
     latitute_location,
     latitute_sigma,
     longitude_location,
     longitude_sigma,
     gaussian_amplitude,
     modified_channels,
 ):
    """ Apply a Gaussain perturbation 
        A = A₀exp( - (x-x₀)²/(2σ₁²) - (y-y₀)²/(2σ₂²) )
        with prescribed A₀, x₀, y₀, σ₁, σ₂
        """
    lat = torch.linspace(-90, 90, x.shape[-2])
    lon = torch.linspace(-180, 180, x.shape[-1])
    lat, lon = torch.meshgrid(lat, lon)

    dt = torch.tensor(time_step.total_seconds()) / 86400.0

    gaussian = dt * gaussian_amplitude * torch.exp(
        -((lon - latitute_location)**2 / (2 * latitute_sigma**2)
        + (lat - longitude_location)**2 / (2 * longitude_sigma**2)))

    for modified_channel in modified_channels:
        index_channel = in_channel_names.index(modified_channel)
        x[:, :, index_channel, :, :] += gaussian.to(device)
    return x


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
        config: EnsembleRun = EnsembleRun.parse_obj(json.loads(config))
    # Otherwise assume parsable obj
    else:
        raise ValueError(
            f"Passed config parameter {config} should be valid file or JSON string"
        )

    # Set up parallel
    DistributedManager.initialize()
    device = DistributedManager().device
    group = torch.distributed.group.WORLD

    logging.info(f"Earth-2 MIP config loaded {config}")
    logging.info(f"Loading model onto device {device}")
    model = get_model(config.weather_model, device=device)
    logging.info(f"Constructing initializer data source")
    perturb = get_initializer(
        model,
        config,
    )
#     model.source = get_source(device, model)
    model.source = partial(
        apply_gaussian_perturbation,
        in_channel_names = model.in_channel_names,
        device=device,
        latitute_location=0.0,
        latitute_sigma=10.0,
        longitude_location=0.0,
        longitude_sigma=10.0,
        gaussian_amplitude=2.0,
        modified_channels=['t850'],
    )
    logging.info("Running inference")
    run_inference(model, config, perturb, group)


if __name__ == "__main__":
    main()
