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
import zipfile

from earth2mip import registry, schema, networks, config, initial_conditions, geometry
from modulus.models.fcn_mip_plugin import _fix_state_dict_keys

# TODO: Update to new arch in Modulus!
import earth2mip.networks.fcnv2 as fcnv2

logger = logging.getLogger(__file__)


def _download_default_package(package):

    model_registry = os.environ["MODEL_REGISTRY"]
    fcn_registry = os.path.join(model_registry, "fcnv2_sm")
    if str(fcn_registry) != str(package.root):
        logger.info("Custom package fcnv2_sm found, aborting default package")
        return

    if not os.path.isdir(package.root):
        logger.info("Downloading FCNv2 small checkpoint, this may take a bit")
        urllib.request.urlretrieve(
            "https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_fcnv2_sm/"
            + "versions/v0.2/files/fcnv2_sm.zip",
            f"{model_registry}/fcnv2_sm.zip",
        )
        # Unzip
        with zipfile.ZipFile(f"{model_registry}/fcnv2_sm.zip", "r") as zip_ref:
            zip_ref.extractall(model_registry)
        # Clean up zip
        os.remove(f"{model_registry}/fcnv2_sm.zip")
    else:
        logger.info("FCNv2 small package already found, skipping download")


def load(package, *, pretrained=True, device="cuda"):
    assert pretrained
    # Download model if needed
    _download_default_package(package)

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

    grid = schema.Grid.grid_721x1440
    channel_set = schema.ChannelSet.var73
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
