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

from earth2mip import grid, regrid


def test_get_regridder():
    src = grid.equiangular_lat_lon_grid(721, 1440)
    dest = grid.equiangular_lat_lon_grid(181, 360)
    try:
        f = regrid.get_regridder(src, dest)
    except FileNotFoundError as e:
        pytest.skip(f"{e}")
    x = torch.ones(1, 1, 721, 1440)
    y = f(x)
    assert y.shape == (1, 1, 181, 360)
    assert torch.allclose(y, torch.ones_like(y))
