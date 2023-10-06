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
FCN adapter from Modulus
"""
from typing import List
import logging
import os
import datetime
import torch
import json
import pathlib

import numpy as np
import onnxruntime as ort
import dataclasses
import modulus

from earth2mip import registry, schema, networks, config, initial_conditions, geometry


def load(package, *, pretrained=True, device="cuda"):
    assert pretrained

    local_center = np.load(package.get("global_means.npy"))
    local_std = np.load(package.get("global_stds.npy"))

    core_model = modulus.Module.from_checkpoint(package.get("fcn.mdlus"))

    grid = schema.Grid.grid_720x1440
    channel_set = schema.ChannelSet.var26
    dt = datetime.timedelta(hours=6)

    inference = networks.Inference(
        core_model,
        channels=None,
        center=local_center,
        scale=local_std,
        grid=grid,
        channel_names=channel_set.list_channels(),
        channel_set=channel_set,
        time_step=dt,
    )
    inference.to(device)
    return inference
