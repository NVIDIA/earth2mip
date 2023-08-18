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

from earth2mip.initial_conditions import cds
from earth2mip import schema
import datetime
import random

import pytest


@pytest.mark.slow
def test_cds_data_source():
    try:
        client = cds.Client()
    except Exception:
        pytest.skip("Could not initialize client")

    time = datetime.datetime(2018, 1, 1)
    channels = ["q1000", "t2m"]
    source = cds.DataSource(channels, client=client)
    dataset = source[time]

    assert source.channel_names == channels
    correct_dims = {"time": 1, "channel": len(channels), "lat": 721, "lon": 1440}
    assert dataset.dims == tuple(correct_dims.keys())
    assert dataset.shape == tuple(correct_dims.values())


def test_make_request(regtest):
    time = datetime.datetime(2018, 1, 1)
    channels = ["q1000", "z1000", "u1000", "t2m", "q10"]
    codes = [cds.parse_channel(c) for c in channels]
    for req in cds._get_cds_requests(codes, time, format="grib"):
        print(req, file=regtest)


def test_parse_channel_with_level():
    channel_level = random.randint(0, 10000)
    channel_string = f"u{channel_level}"
    assert cds.parse_channel(channel_string).level == channel_level


@pytest.mark.parametrize("c", schema.ChannelSet.var73.list_channels())
def test_parse_known_channels(c):
    assert cds.parse_channel(c)
