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

import datetime

import torch

from earth2mip.networks import pangu


class MockPangu(pangu.PanguWeather):
    def __init__(self):
        pass

    def __call__(self, pl, sl):
        return pl, sl


def test_pangu():
    model_6 = pangu.PanguStacked(MockPangu())
    model_24 = pangu.PanguStacked(MockPangu())
    inference = pangu.PanguInference(model_6, model_24)
    t0 = datetime.datetime(2018, 1, 1)
    dt = datetime.timedelta(hours=6)
    x = torch.ones((1, 1, len(inference.in_channel_names), 721, 1440))
    n = 5

    times = []
    for k, (time, y, _) in enumerate(inference(t0, x)):
        if k > n:
            break
        assert y.shape == x.shape[1:]
        assert torch.all(y == x[0])
        times.append(time)

    assert times == [t0 + k * dt for k in range(n + 1)]
