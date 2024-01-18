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

# import this before torch to avoid GLIBC error
import random as rand

import numpy
import pytest
import torch

from earth2mip import config


def get_gpu_count():
    return torch.cuda.device_count()


@pytest.fixture()
def has_registry():
    if not config.MODEL_REGISTRY:
        pytest.skip("MODEL_REGISTRY not configured.")


@pytest.fixture()
def dist():
    from modulus.distributed.manager import DistributedManager

    DistributedManager.initialize()
    return DistributedManager()


@pytest.fixture()
def ngpu():
    return get_gpu_count()


@pytest.fixture
def random():
    rand.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
