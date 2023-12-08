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

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2mip.beta.perturbation import Gaussian


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(2, 4, 16, 16),
        torch.randn(2, 2, 4, 16, 16),
    ],
)
@pytest.mark.parametrize(
    "amplitude",
    [
        1.0,
        0.05,
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_gaussian(x, amplitude, device):

    x = x.to(device)
    coords = OrderedDict([(f"{i}", np.arange(x.shape[i])) for i in range(x.ndim)])

    prtb = Gaussian(amplitude)
    dx = prtb(x, coords)

    assert dx.shape == x.shape
    assert torch.allclose(
        torch.mean(dx), torch.Tensor([0]).to(device), rtol=1e-2, atol=1e-1
    )
    assert torch.allclose(
        torch.std(dx), torch.Tensor([amplitude]).to(device), rtol=1e-2, atol=1e-1
    )
    assert dx.device == x.device
