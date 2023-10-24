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
import logging
import torch
import os

# Set number of GPUs to use to 1
os.environ["WORLD_SIZE"] = "1"
# Set model registry as a local folder
model_registry = os.path.join(os.path.dirname(os.path.realpath(os.getcwd())), "models")
os.makedirs(model_registry, exist_ok=True)
os.environ["MODEL_REGISTRY"] = model_registry

from modulus.distributed.manager import DistributedManager
from earth2mip.inference_ensemble import run_inference, get_initializer
from earth2mip.networks import get_model
from earth2mip.schema import EnsembleRun
from earth2mip.diagnostic import ClimateNet, DiagnosticTimeLoop


def main():
    config_dict = {
        "ensemble_members": 4,
        "noise_amplitude": 0.05,
        "simulation_length": 10,
        "weather_event": {
            "properties": {
                "name": "Globe",
                "start_time": "2018-06-01 00:00:00",
                "initial_condition_source": "cds",
            },
            "domains": [
                {
                    "name": "global",
                    "type": "Window",
                    "diagnostics": [
                        {
                            "type": "raw",
                            "channels": ["u10m", "climnet_tc", "climnet_ar"],
                        }
                    ],
                }
            ],
        },
        "output_path": "../outputs/inference_diagnostic_ensemble",
        "output_frequency": 1,
        "weather_model": "e2mip://fcnv2_sm",
        "seed": 12345,
        "use_cuda_graphs": False,
        "ensemble_batch_size": 1,
        "autocast_fp16": False,
        "perturbation_strategy": "correlated",
        "noise_reddening": 2.0,
    }
    config: EnsembleRun = EnsembleRun.parse_obj(config_dict)
    logging.basicConfig(level=logging.INFO)

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
    # Add diagnostic model to
    logging.info("Loading diagnostic model")
    package = ClimateNet.load_package()
    diagnostic = ClimateNet.load_diagnostic(package)

    model_diagnostic = DiagnosticTimeLoop(diagnostics=[diagnostic], model=model)

    logging.info("Running inference")
    run_inference(model_diagnostic, config, perturb, group)


if __name__ == "__main__":
    main()
