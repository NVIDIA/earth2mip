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

import logging
import os
import shutil

from earth2mip.inference_ensemble import run_inference
from earth2mip.networks import get_model
from earth2mip.schema import EnsembleRun
from earth2mip.initial_conditions import hdf5, cds  # noqa
from datetime import timedelta

logging.basicConfig(level=logging.INFO)


def run_for_a_year(time_loop, output_path):
    dt = time_loop.time_step
    nt = timedelta(days=365) // dt

    config_dict = {
        "simulation_length": nt,
        "weather_event": {
            "properties": {
                "name": "Globe",
                "start_time": "2018-01-02 00:00:00",
                "initial_condition_source": "cds",
            },
            "domains": [
                {
                    "name": "global",
                    "type": "Window",
                    "diagnostics": [{"type": "raw", "channels": ["t2m", "z500"]}],
                }
            ],
        },
        "output_path": output_path,
        "weather_model": "this value is overrided",
        "output_frequency": 4,
    }

    # if you have hdf5 initial condition data available it will be faster
    # data_source = hdf5.DataSource.from_path("s3://lagged-ensemble/validation_data_2018")
    data_source = cds.DataSource(channel_names=time_loop.in_channel_names)
    print("data_source", data_source)
    print("channels", data_source.channel_names)

    config = EnsembleRun.parse_obj(config_dict)

    # Set up parallel
    logging.info("Running inference")
    run_inference(time_loop, config, data_source=data_source)


def main():
    root = "year_runs"
    for model in ["e2mip://dlwp", "e2mip://graphcast", "e2mip://fcnv2_sm"]:
        model_str = model.replace("e2mip://", "")
        output_path = os.path.join(root, model_str)

        if os.path.exists(output_path):
            continue

        tmp_output_path = output_path + ".tmp"
        os.makedirs(tmp_output_path, exist_ok=True)
        time_loop = get_model(model, device="cuda:0")
        run_for_a_year(time_loop, tmp_output_path)
        shutil.move(tmp_output_path, output_path)


if __name__ == "__main__":
    main()
