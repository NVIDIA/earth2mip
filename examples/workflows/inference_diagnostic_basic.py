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
import os
import datetime

# Set number of GPUs to use to 1
os.environ["WORLD_SIZE"] = "1"
# Set model registry as a local folder
model_registry = os.path.join(os.path.dirname(os.path.realpath(os.getcwd())), "models")
os.makedirs(model_registry, exist_ok=True)
os.environ["MODEL_REGISTRY"] = model_registry

from modulus.distributed.manager import DistributedManager
from earth2mip.inference_ensemble import run_basic_inference
from earth2mip.networks import get_model
from earth2mip.initial_conditions import cds
from earth2mip.diagnostic import PrecipitationAFNO, DiagnosticTimeLoop


def main(model_name: str = "e2mip://fcnv2_sm"):

    logging.basicConfig(level=logging.INFO)

    # Set up parallel
    DistributedManager.initialize()
    device = DistributedManager().device

    logging.info(f"Loading model onto device {device}")
    model = get_model(model_name, device=device)

    logging.info("Loading precipitation diagnostic model")
    package = PrecipitationAFNO.load_package()
    diagnostic = PrecipitationAFNO.load_diagnostic(package)

    model_diagnostic = DiagnosticTimeLoop(diagnostics=[diagnostic], model=model)

    logging.info("Constructing initializer data source")
    data_source = cds.DataSource(model.in_channel_names)
    time = datetime.datetime(2018, 4, 4)

    logging.info("Running inference")
    ds = run_basic_inference(
        model_diagnostic,
        n=1,
        data_source=data_source,
        time=time,
    )
    print(ds)


if __name__ == "__main__":
    main()
