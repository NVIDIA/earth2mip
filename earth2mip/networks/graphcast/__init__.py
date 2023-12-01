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
from earth2mip.networks.graphcast import time_loop, implementation

__all__ = ["load_time_loop", "load_time_loop_operational", "load_time_loop_small"]


def _load_time_loop_from_description(
    package,
    checkpoint_path: str,
    resolution: float,
    device="cuda:0",
):
    checkpoint = package.get(os.path.join("params", checkpoint_path))
    dataset = {
        0.25: "source-era5_date-2022-01-01_res-0.25_levels-13_steps-04.nc",
        1.0: "source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc",
    }[resolution]
    dataset_path = package.get(os.path.join("dataset", dataset))
    stats_dir = package.get("stats", recursive=True)
    stepper = implementation.load_graphcast(checkpoint, dataset_path, stats_dir)
    return time_loop.GraphcastTimeLoop(stepper, device=device)


# explicit graphcast versions
def load_time_loop(
    package,
    pretrained=True,
    device="cuda:0",
):
    return _load_time_loop_from_description(
        package=package,
        checkpoint_path="GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz",  # noqa
        resolution=0.25,
        device=device,
    )


def load_time_loop_small(
    package,
    pretrained=True,
    device="cuda:0",
):
    return _load_time_loop_from_description(
        package=package,
        checkpoint_path="GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",  # noqa
        resolution=1.0,
        device=device,
    )


def load_time_loop_operational(
    package,
    pretrained=True,
    device="cuda:0",
):
    return _load_time_loop_from_description(
        package=package,
        checkpoint_path="GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz",  # noqa
        resolution=0.25,
        device=device,
    )
