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
import pathlib
import shutil

import numpy as np
import pytest

from earth2mip.data import CDS


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=1958, month=1, day=31),
        [
            datetime.datetime(year=1971, month=6, day=1, hour=6),
            datetime.datetime(year=2021, month=11, day=23, hour=12),
        ],
    ],
)
@pytest.mark.parametrize("channel", ["tcwv", ["sp", "u500"]])
def test_cds_fetch(time, channel):

    ds = CDS(cache=False)
    data = ds(time, channel)
    shape = data.shape

    if isinstance(channel, str):
        channel = [channel]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(channel)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    assert CDS.available(time[0])


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time",
    [
        [datetime.datetime(year=2023, month=1, day=1)],
    ],
)
@pytest.mark.parametrize("channel", [["z500", "r200"]])
@pytest.mark.parametrize("cache", [True, False])
def test_cds_cache(time, channel, cache):

    ds = CDS(cache=cache)
    data = ds(time, channel)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    # Cahce should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, channel[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=1939, month=2, day=25),
        datetime.datetime(year=1, month=1, day=1, hour=13, minute=1),
        datetime.datetime.now(),
    ],
)
@pytest.mark.parametrize("channel", ["mpl"])
def test_cds_available(time, channel):
    assert not CDS.available(time)
    with pytest.raises(ValueError):
        ds = CDS()
        ds(time, channel)
