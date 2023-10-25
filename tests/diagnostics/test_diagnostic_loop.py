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
import pytest
import datetime
import torch
import earth2mip.schema as schema
from earth2mip.networks import Inference
from earth2mip.diagnostic import DiagnosticTimeLoop, WindSpeed


class Identity(torch.nn.Module):
    def forward(self, x):
        return x + 0.01


@pytest.parametrize
def test_inference_run_with_restart():
    network = Identity()
    center = [0, 0, 0, 0]
    scale = [1, 1, 1, 1]

    # batch, time_levels, channels, y, x
    x = torch.zeros([1, 1, 4, 5, 6])
    model = Inference(
        network,
        center=center,
        scale=scale,
        grid=schema.Grid.grid_720x1440,
        channel_names=["u10m", "v10m", "tcwv", "msp"],
    )

    diagWS = WindSpeed.load_diagnostic(level="10m", grid=schema.Grid.grid_720x1440)
    diag_model = DiagnosticTimeLoop(diagnostics=[diagWS], model=model, concat=True)

    time = datetime.datetime(2018, 1, 1)
    for k, (data, time, _) in enumerate(diag_model(time, x)):
        print(data.shape)
