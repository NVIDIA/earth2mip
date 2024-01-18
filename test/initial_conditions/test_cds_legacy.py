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
import random

import pytest

from earth2mip.initial_conditions import cds


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
    assert dataset.shape == (len(channels), 721, 1440)


def test_make_request(regtest):
    time = datetime.datetime(2018, 1, 1)
    channels = ["q1000", "z1000", "u1000", "t2m", "v100m"]
    codes = [cds.parse_channel(c) for c in channels]
    for req in cds._get_cds_requests(codes, time, format="grib"):
        print(req, file=regtest)


def test_parse_channel_with_level():
    channel_level = random.randint(0, 10000)
    channel_string = f"u{channel_level}"
    assert cds.parse_channel(channel_string).level == channel_level


@pytest.mark.parametrize("c", ["t850", "t2m", "u10", "z100"])
def test_parse_known_channels(c: str):
    assert cds.parse_channel(c)
