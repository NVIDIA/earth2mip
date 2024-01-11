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

import numpy as np
import pytest
import test_hdf5
import torch

from earth2mip import config, grid, initial_conditions, schema
from earth2mip.initial_conditions import cds, hdf5
from earth2mip.initial_conditions.base import DataSource


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

    if source == schema.InitialConditionSource.cds:
        try:
            cds.Client()
        except Exception:
            pytest.skip("Could not initialize client")

    ds = initial_conditions.get_data_source(
        channel_names=["t850", "t2m"],
        initial_condition_source=source,
    )
    assert isinstance(ds, DataSource)


@pytest.mark.parametrize("n", [1, 2])
def test_get_initial_conditions_for_model_hdf5(tmp_path, n):
    class Model:
        in_channel_names = ["t850", "t2m"]
        n_history_levels = n
        history_time_step = datetime.timedelta(hours=6)
        grid = grid.equiangular_lat_lon_grid(721, 1440)
        device = "cpu"
        dtype = torch.float

    time = datetime.datetime(2018, 1, 2)
    test_hdf5.create_hdf5(tmp_path, time.year, 10, Model.grid, Model.in_channel_names)
    data_source = hdf5.DataSource.from_path(tmp_path.as_posix())

    x = initial_conditions.get_initial_condition_for_model(Model, data_source, time)
    assert (
        x.shape
        == (1, Model.n_history_levels, len(Model.in_channel_names)) + Model.grid.shape
    )


@pytest.mark.parametrize("n", [1, 2])
def test_get_initial_conditions_for_model_history_correct_order(n):
    """Tests that the history dim is stacked increasing order in time"""

    class Model:
        in_channel_names = ["t850", "t2m"]
        n_history_levels = n
        history_time_step = datetime.timedelta(hours=6)
        grid = grid.equiangular_lat_lon_grid(721, 1440)
        device = "cpu"
        dtype = torch.float

    shape = (len(Model.in_channel_names), *Model.grid.shape)
    time = datetime.datetime(2018, 1, 1)

    class DataSource(dict):
        channel_names = Model.in_channel_names
        grid = Model.grid

    data_source = DataSource()
    dt = Model.history_time_step
    for i in range(Model.n_history_levels):
        data_source[time - i * dt] = np.full(shape, fill_value=-i)

    x = initial_conditions.get_initial_condition_for_model(Model, data_source, time)
    for i in range(Model.n_history_levels):
        assert x[0, -i - 1, 0, 0, 0] == -i
