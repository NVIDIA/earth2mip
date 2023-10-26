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
import os
from functools import partial
import torch
import datetime
from modulus.distributed.manager import DistributedManager
from earth2mip.inference_ensemble import run_basic_inference
from earth2mip.networks import get_model
from earth2mip.initial_conditions import cds


def gaussian_source(
    x: torch.Tensor,
    in_channel_names: List[str],
    lat: torch.Tensor,
    lon: torch.Tensor,
    amplitude: float,
    channel_to_perturb: str,
    latitute_location:float =0,
    latitute_sigma:float =50,
    longitude_location: float =0,
    longitude_sigma=10,
) -> torch.Tensor:
    """Apply a Gaussain nudge of the from
    A = A₀exp( - (x-x₀)²/(2σ₁²) - (y-y₀)²/(2σ₂²) )
    with prescribed A₀, x₀, y₀, σ₁, σ₂
    to user prescribed variable(s)
    
    Args:
        x: the input state to modify. Shape (B, C, nlat, nlon)
        lat: the latitude shape (lat,)
        lon: the longitude in degrees east. 0<= lon < 360, shape is (nlon)
        amplitude: the size of the perturbation in X/ seconds, where X is the units of `channel_to_perturb`
    
    Returns:
       source: the source evaluated in units X / seconds. where X is the units of `channel_to_perturb`
    """
    lat, lon = torch.meshgrid(lat, lon)
    source = torch.zeros_like(x)
    blob = amplitude * torch.exp(
            -(
                (lon - latitute_location) ** 2 / (2 * latitute_sigma**2)
                + (lat - longitude_location) ** 2 / (2 * longitude_sigma**2)
            )
        )
    )
    index = in_channels.index(channel_to_perturb)
    source[:, index] = blob
    return source


def main():
    device = DistributedManager().device
    model = get_model("pangu", device=device)
    model.source = partial(
        apply_gaussian_perturbation,
        in_channel_names=model.in_channel_names,
        device=device,
        latitute_location=0.0,
        latitute_sigma=5.0,
        longitude_location=0.0,
        longitude_sigma=5.0,
        gaussian_amplitude=10.0,
        modified_channels=["t850"],
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