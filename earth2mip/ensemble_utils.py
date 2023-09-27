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
import numpy as np  # noqa
import h5py  # noqa
import os  # noqa

from einops import rearrange  # noqa
from earth2mip import schema  # noqa
import torch_harmonics as th  # noqa
from earth2mip.networks import Inference  # noqa
from datetime import datetime
from timeit import default_timer  # noqa
from typing import Union


def apply_gaussian_perturbation(
    x,
    time_step,
    channel_set,
    device,
    latitute_location,
    latitute_sigma,
    longitude_location,
    longitude_sigma,
    gaussian_amplitude,
    modified_channels,
):
    shape = x.shape[-2:]
    lat = torch.linspace(-90, 90, shape[-2])
    lon = torch.linspace(-180, 180, shape[-1])
    lon, lat = torch.meshgrid(lon, lat)

    dt = torch.tensor(time_step.total_seconds()) / 3600.0

    gaussian = dt * gaussian_amplitude * torch.exp(
        -((lon - latitute_location)**2 / (2 * latitute_sigma**2)
          + (lat - longitude_location)**2 / (2 * longitude_sigma**2)))

    gaussian = gaussian.transpose(-1, -2).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    gaussian = gaussian.expand(x.shape[0], x.shape[1], 1, x.shape[-2], x.shape[-1])
    channel_list = channel_set.list_channels()
    for modified_channel in modified_channels:
        index_channel = channel_list.index(modified_channel)
        x[:, :, index_channel, :, :] += gaussian.squeeze(2).to(device)
    return x


def generate_noise_correlated(shape, *, reddening, device, noise_amplitude):
    return noise_amplitude * brown_noise(shape, reddening).to(device)


def brown_noise(shape, reddening=2):
    noise = torch.normal(torch.zeros(shape), torch.ones(shape))

    x_white = torch.fft.fft2(noise)
    S = (
        torch.abs(torch.fft.fftfreq(noise.shape[-2]).reshape(-1, 1)) ** reddening
        + torch.abs(torch.fft.fftfreq(noise.shape[-1])) ** reddening
    )

    S = torch.where(S == 0, 0, 1 / S)
    S = S / torch.sqrt(torch.mean(S**2))

    x_shaped = x_white * S
    noise_shaped = torch.fft.ifft2(x_shaped).real

    return noise_shaped


def generate_bred_vector(
    x: torch.Tensor,
    model: Inference,
    noise_amplitude: float = 0.15,
    time: Union[datetime, None] = None,
    integration_steps: int = 40,
    inflate=False,
):
    # Assume x has shape [ENSEMBLE, TIME, CHANNEL, LAT, LON]
    x0 = x[:1]

    # Get control forecast
    for data in model.run_steps(x0, n=1, normalize=False, time=time):
        xd = data

    # Unsqueeze if time has been collapsed.
    if xd.ndim != x0.ndim:
        xd = xd.unsqueeze(1)

    dx = noise_amplitude * torch.randn(x.shape, device=x.device, dtype=x.dtype)
    for _ in range(integration_steps):
        x1 = x + dx
        for data in model.run_steps(x1, n=1, normalize=False, time=time):
            x2 = data

        # Unsqueeze if time has been collapsed.
        if x2.ndim != x1.ndim:
            x2 = x2.unsqueeze(1)
        dx = x2 - xd

        if inflate:
            dx += noise_amplitude * (dx - dx.mean(dim=0))

    gamma = torch.norm(x) / torch.norm(x + dx)
    return noise_amplitude * dx * gamma
