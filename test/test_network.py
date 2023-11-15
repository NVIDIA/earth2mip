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

import numpy as np
import torch
import torch.nn

import earth2mip.grid
from earth2mip import networks


class Identity(torch.nn.Module):
    def forward(self, x):
        return x + 0.01


def test_inference_run_with_restart():
    model = Identity()
    center = [0, 0]
    scale = [1, 1]

    # batch, time_levels, channels, y, x
    x = torch.zeros([1, 1, 2, 5, 6])
    model = networks.Inference(
        model,
        center=center,
        scale=scale,
        grid=earth2mip.grid.equiangular_lat_lon_grid(
            720, 1440, includes_south_pole=False
        ),
        channel_names=["a", "b"],
    )

    step1 = []
    time = datetime.datetime(2018, 1, 1)
    for k, (_, state, restart) in enumerate(model(time, x)):
        step1.append(restart)
        if k == 3:
            break
    assert len(step1) == 4

    # start run from 50% done
    for k, (_, final_state, _) in enumerate(model(time, x=None, restart=step1[1])):
        if k == 2:
            break

    np.testing.assert_array_equal(final_state.numpy(), state.numpy())
