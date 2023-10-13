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

from earth2mip import networks, schema
import torch
import torch.nn
import numpy as np


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
        grid=schema.Grid.grid_720x1440,
        channel_names=["a", "b"],
    )

    step1 = []
    for _, state, restart in model.run_steps_with_restart(x, 3):
        step1.append(restart)
    assert len(step1) == 4

    # start run from 50% done
    for _, final_state, _ in model.run_steps_with_restart(n=2, **step1[1]):
        pass

    np.testing.assert_array_equal(final_state.numpy(), state.numpy())
