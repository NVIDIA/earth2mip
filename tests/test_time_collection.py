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

import os
import json
import numpy as np
import torch
import subprocess
import xarray
from earth2mip.datasets.hindcast import open_forecast

import pytest

DIR = os.path.dirname(__file__)
os.environ["PYTHONPATH"] = DIR + ":" + os.getenv("PYTHONPATH", ":")


def create_model_package(tmp_path):
    package_dir = tmp_path / "mock_package"
    package_dir.mkdir()

    # Create metadata.json
    metadata = {
        "architecture_entrypoint": "mock_plugin:load",
        "n_history": 0,
        "channel_set": "73var",
        "grid": "721x1440",
        "in_channels": list(range(73)),
        "out_channels": list(range(73)),
    }
    with open(package_dir.joinpath("metadata.json"), "w") as f:
        json.dump(metadata, f)

    # Create numpy arrays
    global_means = np.zeros((1, 73, 1, 1))
    global_stds = np.ones((1, 73, 1, 1))
    np.save(package_dir.joinpath("global_means.npy"), global_means)
    np.save(package_dir.joinpath("global_stds.npy"), global_stds)

    return package_dir


@pytest.mark.slow
@pytest.mark.xfail
def test_time_collection(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs gpu and data")

    torch.init_process_group()
    model_package = create_model_package(tmp_path)

    config = os.path.join(DIR, "configs/medium-test.json")
    model = f"file://{model_package.as_posix()}"

    root = str(tmp_path / "test")
    subprocess.check_call(["python3", "-m", "earth2mip.make_job", model, config, root])
    subprocess.check_call(
        [
            "torchrun",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(min([torch.cuda.device_count(), 2])),
            "-m",
            "earth2mip.time_collection",
            root,
        ]
    )

    ds = open_forecast(root, group="mean.zarr")
    assert isinstance(ds, xarray.Dataset)
