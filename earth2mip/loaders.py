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

from typing import Protocol

import torch


class LoaderProtocol(Protocol):
    def __call__(self, package, pretrained=True) -> None:
        return


def torchscript(package, pretrained=True):
    """
    load a checkpoint into a model
    """
    p = package.get("scripted_model.pt")
    import json

    config = package.get("config.json")
    with open(config) as f:
        config = json.load(f)

    model = torch.jit.load(p)

    if config["add_zenith"]:
        import numpy as np

        from earth2mip.networks import CosZenWrapper

        lat = 90 - np.arange(721) * 0.25
        lon = np.arange(1440) * 0.25
        model = CosZenWrapper(model, lon, lat)

    return model
