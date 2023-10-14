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
from earth2mip import initial_conditions, schema
from earth2mip.initial_conditions.base import DataSource
from earth2mip import config
import pytest


@pytest.mark.parametrize("source", list(schema.InitialConditionSource))
def test_get_data_source(
    source: schema.InitialConditionSource, monkeypatch: pytest.MonkeyPatch
):

    if (
        source
        in [schema.InitialConditionSource.era5, schema.InitialConditionSource.hrmip]
        and not config.ERA5_HDF5
    ):
        pytest.skip(f"Need HDF5 data to test {source}")

    ds = initial_conditions.get_data_source(
        n_history=0,
        channel_names=["t850", "t2m"],
        grid=schema.Grid.grid_721x1440,
        initial_condition_source=source,
    )
    assert isinstance(ds, DataSource)
