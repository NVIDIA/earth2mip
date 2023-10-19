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


def generate_model_noise_correlated(x,
                                    time_step,
                                    reddening,
                                    device,
                                    noise_injection_amplitude,
                                    ):
    shape = x.shape
    dt = torch.tensor(time_step.total_seconds()) / 3600.0
    noise = noise_injection_amplitude * dt * brown_noise(shape, reddening).to(device)
    return x * (1.0 + noise)


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
    logging.info("Constructing initializer data source")
    perturb = get_initializer(
        model,
        config,
    )
    model.source = partial(
        generate_model_noise_correlated,
        reddening=2.0,
        device=device,
        noise_injection_amplitude=0.003,
    )
    logging.info("Running inference")
    run_inference(model, config, perturb, group)


if __name__ == "__main__":
    main()
