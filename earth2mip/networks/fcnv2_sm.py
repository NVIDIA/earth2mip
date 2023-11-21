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

"""
FCN v2 Small adapter

This model is an outdated version of FCN v2 (SFNO), a more recent one is present in Modulus.
"""
import datetime
import logging
import pathlib

import numpy as np
import torch

import earth2mip.grid

# TODO: Update to new arch in Modulus!
import earth2mip.networks.fcnv2 as fcnv2
from earth2mip import networks

logger = logging.getLogger(__file__)

CHANNELS = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "r50",
    "r100",
    "r150",
    "r200",
    "r250",
    "r300",
    "r400",
    "r500",
    "r600",
    "r700",
    "r850",
    "r925",
    "r1000",
]


def _fix_state_dict_keys(state_dict, add_module=False):
    """Add or remove 'module.' from state_dict keys

    Parameters
    ----------
    state_dict : Dict
        Model state_dict
    add_module : bool, optional
        If True, will add 'module.' to keys, by default False

    Returns
    -------
    Dict
        Model state_dict with fixed keys
    """
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if add_module:
            new_key = "module." + key
        else:
            new_key = key.replace("module.", "")
        fixed_state_dict[new_key] = value
    return fixed_state_dict


def load(package, *, pretrained=True, device="cuda"):
    assert pretrained  # noqa

    config_path = pathlib.Path(__file__).parent / "fcnv2" / "sfnonet.yaml"
    params = fcnv2.YParams(config_path.as_posix(), "sfno_73ch")
    params.img_crop_shape_x = 721
    params.img_crop_shape_y = 1440
    params.N_in_channels = 73
    params.N_out_channels = 73

    core_model = fcnv2.FourierNeuralOperatorNet(params).to(device)

    local_center = np.load(package.get("global_means.npy"))
    local_std = np.load(package.get("global_stds.npy"))

    weights_path = package.get("weights.tar")
    weights = torch.load(weights_path, map_location=device)
    fixed_weights = _fix_state_dict_keys(weights["model_state"], add_module=False)
    core_model.load_state_dict(fixed_weights)

    grid = earth2mip.grid.equiangular_lat_lon_grid(721, 1440)
    dt = datetime.timedelta(hours=6)

    inference = networks.Inference(
        core_model,
        center=local_center,
        scale=local_std,
        grid=grid,
        channel_names=CHANNELS,
        time_step=dt,
    )
    inference.to(device)
    return inference
