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

import pytest

from earth2mip.initial_conditions.ifs import DataSource, _get_filename


def test__get_filename():
    expected = "20230310/00z/0p4-beta/oper/20230310000000-0h-oper-fc.grib2"
    time = datetime.datetime(2023, 3, 10, 0)
    assert _get_filename(time, "0h") == expected


@pytest.mark.slow
@pytest.mark.xfail
def test_get():
    # uses I/O and old ICs are not available forever.
    time = datetime.datetime.utcnow()
    last_time = datetime.datetime(time.year, time.month, time.day)

    data = DataSource(["t2m"])
    ds = data[last_time]
    assert ds.shape == (1, *data.grid.shape)
