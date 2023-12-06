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
import torch

from earth2mip import grid
from earth2mip.diagnostic import WindSpeed


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("grid", [grid.equiangular_lat_lon_grid(32, 64)])
def test_wind_speed(device, grid):
    model = WindSpeed.load_diagnostic(None, level="10m", grid=grid)
    x = torch.randn(2, len(model.in_channel_names), len(grid.lat), len(grid.lon)).to(
        device
    )
    out = model(x)
    assert torch.allclose(torch.sqrt(x[:, :1] ** 2 + x[:, 1:] ** 2), out)
