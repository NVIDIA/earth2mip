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

import pathlib
import numpy as np
import datetime

import torch
from earth2mip import schema, model_registry, networks
from earth2mip.model_registry import Package

import hashlib


def md5_checksum(x, precision):
    x_rounded = np.round_(x, precision)
    x_string = x_rounded.tostring()
    md5 = hashlib.md5(x_string)
    checksum = md5.hexdigest()
    return checksum


def _mock_registry_with_metadata(metadata, model_name, tmp_path):

    root = tmp_path / model_name
    root.mkdir()

    registry = model_registry.ModelRegistry(tmp_path.as_posix())
    registry.put_metadata(model_name, metadata)

    # save stats
    def save_ones(path):
        out = np.ones((len(metadata.in_channels),))
        np.save(path, out)

    save_ones(registry.get_scale_path(model_name))
    save_ones(registry.get_center_path(model_name))

    return registry


def test_pickle(tmp_path: pathlib.Path):

    model_name = "model"
    # use a baseline AFNO model as a mock
    model = torch.nn.Conv2d(3, 3, 1)

    # Save the model to the registry with appropriate metadata
    metadata = schema.Model(
        architecture="pickle",
        n_history=0,
        channel_set=schema.ChannelSet.var34,
        grid=schema.Grid.grid_720x1440,
        in_channels=list(range(3)),
        out_channels=list(range(3)),
    )

    registry = _mock_registry_with_metadata(metadata, model_name, tmp_path)

    # save model weights
    torch.save(model, registry.get_weight_path(model_name))

    # make sure it works
    loaded = networks.get_model(model_name, registry)
    assert loaded.in_channel_names == [
        metadata.channel_set.list_channels()[i] for i in metadata.in_channels
    ]

    # only do following if cuda enabled, it's too slow on the cpu
    n_history = 0
    ic = torch.ones(1, n_history + 1, len(metadata.in_channels), 2, 4)
    time = datetime.datetime(1, 1, 1)

    for k, (_, b, _) in enumerate(loaded(time, ic)):
        if k > 10:
            break

    assert b.shape == ic[:, 0].shape


def MockLoader(package, pretrained):
    assert pretrained
    return torch.nn.Linear(3, 3)


def test_get_model_architecture_entrypoint(tmp_path):
    registry = model_registry.ModelRegistry(tmp_path.as_posix())
    metadata = schema.Model(
        architecture_entrypoint="tests.test_models:MockLoader",
        n_history=0,
        channel_set=schema.ChannelSet.var34,
        grid=schema.Grid.grid_720x1440,
        in_channels=list(range(3)),
        out_channels=list(range(3)),
    )

    model_name = "model"
    registry = _mock_registry_with_metadata(metadata, model_name, tmp_path)
    model = networks.get_model(model_name, registry)
    assert isinstance(model, torch.nn.Module)


class MyTestInference:
    def __init__(self, package, device, **kwargs):
        self.kwargs = kwargs
        self.device = device


def test__load_package_entrypoint():
    package = Package("", seperator="/")
    metadata = schema.Model(
        entrypoint=schema.InferenceEntrypoint(
            name="tests.test_models:MyTestInference", kwargs=dict(param=1)
        )
    )
    obj = networks._load_package(package, metadata, device="cpu")
    assert isinstance(obj, MyTestInference)
    assert obj.kwargs == metadata.entrypoint.kwargs
    assert obj.device == "cpu"
