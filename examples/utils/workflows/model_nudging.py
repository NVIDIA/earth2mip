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

# %%
import datetime
import os
from functools import partial

import torch
from modulus.distributed.manager import DistributedManager

from earth2mip.inference_ensemble import run_basic_inference
from earth2mip.initial_conditions import cds
from earth2mip.networks import get_model


def gaussian_source(
    x: torch.Tensor,
    time,
    model,
    amplitudes: list[float],
    channels_to_perturb: list[str],
    latitute_location: float = 0,
    latitute_sigma: float = 50,
    longitude_location: float = 0,
    longitude_sigma: float = 10,
) -> torch.Tensor:
    """Apply a Gaussain nudge of the from
    A = A₀exp( - (x-x₀)²/(2σ₁²) - (y-y₀)²/(2σ₂²) )
    with prescribed A₀, x₀, y₀, σ₁, σ₂
    to user prescribed variable(s)

    Args:
        x: the input state to modify. Shape (B, C, nlat, nlon)
        lat: the latitude shape (lat,)
        lon: the longitude in degrees east. 0<= lon < 360, shape is (nlon)
        amplitude: the size of the perturbation in X / day,
        where X is the units of `channel_to_perturb`

    Returns:
       source: the source evaluated in units X / seconds.
       where X is the units of `channel_to_perturb`
    """
    lat, lon = torch.meshgrid(
        torch.tensor(model.grid.lat),
        torch.tensor(model.grid.lon),
    )
    source = torch.zeros_like(x)
    n = len(amplitudes)
    amplitudes = torch.tensor(amplitudes) / 86400
    amplitudes_ = amplitudes.view(n, 1, 1) * torch.zeros(n, lat.size(0), lat.size(1))
    blob = amplitudes_ * torch.exp(
        -(
            (lon - latitute_location) ** 2 / (2 * latitute_sigma**2)
            + (lat - longitude_location) ** 2 / (2 * longitude_sigma**2)
        )
    )
    channel_mask = torch.tensor(
        [channel in channels_to_perturb for channel in model.in_channel_names],
        dtype=torch.bool,
    )
    source[..., channel_mask, :, :] = blob.unsqueeze(0).to(x.device)
    return source


def main():
    device = DistributedManager().device
    model = get_model("e2mip://pangu", device=device)
    model.source = partial(
        gaussian_source,
        model=model,
        amplitudes=[10.0, 5.0],
        channels_to_perturb=["t850", "t500"],
        latitute_location=0.0,
        latitute_sigma=5.0,
        longitude_location=0.0,
        longitude_sigma=5.0,
    )
    time = datetime.datetime(2018, 1, 1)
    data_source = cds.DataSource(model.in_channel_names)
    ds = run_basic_inference(model, 28, data_source, time)
    print(ds)

    # %% Post-process
    import matplotlib.pyplot as plt

    output = "./examples/outputs/perturbation_example"
    os.makedirs(output, exist_ok=True)

    arr = ds.sel(channel="u850").values
    fig, axs = plt.subplots(1, 13, figsize=(13 * 5, 5))
    for i in range(13):
        axs[i].imshow(arr[i, 0])
    plt.savefig(os.path.join(output, "u850_field.png"), bbox_inches="tight")


if __name__ == "__main__":
    main()
