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
import pytest
import torch

from earth2mip.diagnostic.utils import filter_channels


@pytest.mark.parametrize("device", ["cpu"])
def test_filter_channels(device):

    input = torch.randn(1, 1, 3, 2, 2).to(device)
    output = filter_channels(input, ["a", "b", "c"], ["a", "b"])
    assert torch.allclose(input[:, :, :2], output)

    input = torch.randn(3, 5, 5).to(device)
    output = filter_channels(input, ["a", "b", "c"], ["c"])
    assert torch.allclose(input[2:], output)
