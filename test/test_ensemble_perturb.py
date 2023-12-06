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

from earth2mip import networks
from earth2mip.ensemble_utils import generate_bred_vector, generate_noise_correlated
from earth2mip.schema import Grid


@pytest.mark.slow
def test_generate_noise_correlated():
    torch.manual_seed(0)
    shape = (2, 34, 32, 64)
    noise = generate_noise_correlated(
        shape=shape, reddening=2.0, noise_amplitude=0.1, device="cpu"
    )
    assert tuple(noise.shape) == tuple(shape)
    assert torch.mean(noise) < torch.tensor(1e-09).to()


class Dummy(torch.nn.Module):
    def forward(self, x, time):
        return 2.5 * torch.abs(x) * (1 - torch.abs(x))


def test_bred_vector():
    device = "cpu"
    model = Dummy().to(device)
    initial_time = datetime.datetime(2018, 1, 1)
    center = [0, 0]
    scale = [1, 1]

    # batch, time_levels, channels, y, x
    x = torch.rand([4, 1, 2, 5, 6], device=device)
    model = networks.Inference(
        model,
        center=center,
        scale=scale,
        grid=Grid.grid_720x1440,
        channel_names=["a", "b"],
    ).to(device)

    noise_amplitude = 0.01
    noise = generate_bred_vector(
        x,
        model,
        noise_amplitude=noise_amplitude,
        time=initial_time,
        integration_steps=20,
        inflate=False,
    )
    assert noise.device == x.device
    assert noise.shape == x.shape
    assert not torch.any(torch.isnan(noise))

    noise = generate_bred_vector(
        x,
        model,
        noise_amplitude=noise_amplitude,
        time=initial_time,
        integration_steps=20,
        inflate=True,
    )
    assert noise.device == x.device
    assert noise.shape == x.shape
    assert not torch.any(torch.isnan(noise))
