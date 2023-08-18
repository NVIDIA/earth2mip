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

import torch
import pytest
from earth2mip import regrid
from earth2mip import schema


def test_get_regridder():
    src = schema.Grid.grid_721x1440
    dest = schema.Grid.s2s_challenge
    try:
        f = regrid.get_regridder(src, dest)
    except FileNotFoundError as e:
        pytest.skip(f"{e}")
    x = torch.ones(1, 1, 721, 1440)
    y = f(x)
    assert y.shape == (1, 1, 121, 240)
    assert torch.allclose(y, torch.ones_like(y))
