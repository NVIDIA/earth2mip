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

import argparse
import hashlib

import numpy as np
import pytest
import torch

from earth2mip import _cli_utils, model_registry, networks, schema
from earth2mip.model_registry import Package


def md5_checksum(x, precision):
    x_rounded = np.round_(x, precision)
    x_string = x_rounded.tostring()
    md5 = hashlib.md5(x_string)  # noqa: S324
    checksum = md5.hexdigest()
    return checksum


def _mock_registry_with_metadata(metadata, model_name, tmp_path):

    root = tmp_path / model_name
    root.mkdir()

    registry = model_registry.ModelRegistry(tmp_path.as_posix())
    registry.put_metadata(model_name, metadata)

    # save stats
    def save_ones(path):
        out = np.ones((len(metadata.in_channels_names),))
        np.save(path, out)

    save_ones(registry.get_scale_path(model_name))
    save_ones(registry.get_center_path(model_name))

    return registry


def MockLoader(package, pretrained):
    assert pretrained
    return torch.nn.Linear(3, 3)


def test_get_model_architecture_entrypoint(tmp_path):
    registry = model_registry.ModelRegistry(tmp_path.as_posix())
    metadata = schema.Model(
        architecture_entrypoint="test.test_models:MockLoader",
        n_history=0,
        grid=schema.Grid.grid_720x1440,
        in_channels_names=["a", "b", "c"],
        out_channels_names=["a", "b", "c"],
    )

    model_name = "model"
    registry = _mock_registry_with_metadata(metadata, model_name, tmp_path)
    model = networks.get_model(model_name, registry)
    assert isinstance(model, torch.nn.Module)


def test_get_model_uses_metadata(tmp_path):
    registry = model_registry.ModelRegistry(tmp_path.as_posix())
    model_name = "model"
    model = networks.get_model(model_name, registry, metadata=metadata_with_entrypoint)
    assert isinstance(model, MyTestInference)


@pytest.mark.parametrize("required", [True, False])
def test__cli_utils(tmp_path, required):
    path = tmp_path / "meta.json"

    with path.open("w") as f:
        f.write(metadata_with_entrypoint.json())

    parser = argparse.ArgumentParser()
    _cli_utils.add_model_args(parser, required=required)
    model_args = ["model"] if required else ["--model", "unused"]
    args = parser.parse_args(model_args + ["--model-metadata", path.as_posix()])
    loop = _cli_utils.model_from_args(args, device="cpu")
    assert isinstance(loop, MyTestInference)


class MyTestInference:
    def __init__(self, package, device, **kwargs):
        self.kwargs = kwargs
        self.device = device


metadata_with_entrypoint = schema.Model(
    entrypoint=schema.InferenceEntrypoint(
        name="test.test_models:MyTestInference", kwargs=dict(param=1)
    )
)


def test__load_package_entrypoint():
    package = Package("", seperator="/")
    obj = networks._load_package(package, metadata_with_entrypoint, device="cpu")
    assert isinstance(obj, MyTestInference)
    assert obj.kwargs == metadata_with_entrypoint.entrypoint.kwargs
    assert obj.device == "cpu"
