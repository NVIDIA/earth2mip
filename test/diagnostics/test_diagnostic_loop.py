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
import datetime

import pytest
import torch

from earth2mip.diagnostic import DiagnosticTimeLoop, WindSpeed
from earth2mip.diagnostic.utils import filter_channels
from earth2mip.grid import equiangular_lat_lon_grid
from earth2mip.networks import Inference


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("diag_concat", [True, False])
def test_diagnostic_loop_ws(device, diag_concat):
    network = Identity().to(device)
    center = [0, 0, 0, 0]
    scale = [1, 1, 1, 1]

    # batch, time_levels, channels, y, x
    grid = equiangular_lat_lon_grid(8, 16)
    x = torch.rand([1, 1, 4, 8, 16])
    model = Inference(
        network,
        center=center,
        scale=scale,
        grid=grid,
        channel_names=["u10m", "v10m", "tcwv", "msp"],
    )

    diagWS = WindSpeed(level="10m", grid=grid)
    diag_model = DiagnosticTimeLoop(
        diagnostics=[diagWS], model=model, concat=diag_concat
    )

    time = datetime.datetime(2018, 1, 1)
    for k, (time, data, _) in enumerate(diag_model(time, x)):
        assert data.shape == (1, len(diag_model.out_channel_names), 8, 16)
        ws = filter_channels(data, diag_model.out_channel_names, ["ws10m"])
        ws_truth = torch.sqrt(torch.sum(x[:, :, :2] ** 2, dim=2))
        assert torch.allclose(ws_truth, ws)

        if k > 3:
            break
