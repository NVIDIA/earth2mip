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

import numpy as np
import pytest
import torch

from earth2mip import geometry
from earth2mip.weather_events import Window


@pytest.mark.xfail
def test_select_space():
    domain = Window(name="latitude", lat_min=-15, lat_max=15, diagnostics=[])
    lat = np.array([-20, 0, 20])
    lon = np.array([0, 1, 2])
    data = torch.ones((1, 1, len(lat), len(lon))).float()
    lat, lon, output = geometry.select_space(data, lat, lon, domain)
    assert tuple(output.shape[2:]) == (len(lat), len(lon))
    assert np.all(np.abs(lat) <= 15)


@pytest.mark.xfail
def test_get_bounds_window():
    domain = Window(name="latitude", lat_min=-15, lat_max=15, diagnostics=[])
    lat = np.array([-20, 0, 20])
    lon = np.array([0, 1, 2])
    lat_sl, _ = geometry.get_bounds_window(domain, lat, lon)
    assert lat[lat_sl].shape == (1,)
    assert lat[lat_sl][0] == 0
