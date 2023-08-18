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

from earth2mip.initial_conditions import get
from earth2mip.initial_conditions import cds
from earth2mip import schema
import datetime

import pytest


@pytest.mark.slow
@pytest.mark.xfail
def test_get():
    # uses I/O and old ICs are not available forever.
    time = datetime.datetime(2023, 3, 10, 0)
    dataset = get(
        0, time, schema.ChannelSet.var34, source=schema.InitialConditionSource.cds
    )

    # check dims
    correct_dims = {"time": 1, "channel": 34, "lat": 721, "lon": 1440}
    assert dataset.dims == tuple(correct_dims.keys())
    assert dataset.shape == tuple(correct_dims.values())


@pytest.mark.slow
def test_cds_data_source():
    time = datetime.datetime(2018, 1, 1)
    channels = ["q1000"]
    source = cds.DataSource(channels)
    dataset = source[time]

    assert source.channel_names == channels
    correct_dims = {"time": 1, "channel": len(channels), "lat": 721, "lon": 1440}
    assert dataset.dims == tuple(correct_dims.keys())
    assert dataset.shape == tuple(correct_dims.values())
