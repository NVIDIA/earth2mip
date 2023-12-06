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

from earth2mip.diagnostic.climate_net import ClimateNet


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_climate_net_package(device):

    package = ClimateNet.load_package()
    model = ClimateNet.load_diagnostic(package, device)

    x = torch.randn(
        1, len(model.in_channel_names), len(model.in_grid.lat), len(model.in_grid.lon)
    ).to(device)
    out = model(x)
    assert out.size() == (
        1,
        len(model.out_channel_names),
        len(model.out_grid.lat),
        len(model.out_grid.lon),
    )
