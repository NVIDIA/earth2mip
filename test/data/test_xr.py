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

import numpy as np
import pytest
import xarray as xr

from earth2mip.beta.data import DataArrayFile, DataSetFile


@pytest.fixture
def foo_data_array():

    time = [
        datetime.datetime(year=2018, month=1, day=1),
        datetime.datetime(year=2018, month=2, day=1),
        datetime.datetime(year=2018, month=3, day=1),
    ]
    variable = ["u10m", "v10m", "t2m"]

    da = xr.DataArray(
        data=np.random.randn(len(time), len(variable), 8, 16),
        dims=["time", "variable", "lat", "lon"],
        coords={
            "time": time,
            "variable": variable,
        },
    )
    return da


@pytest.fixture
def foo_data_set():

    time = [
        datetime.datetime(year=2018, month=1, day=1),
        datetime.datetime(year=2018, month=2, day=1),
        datetime.datetime(year=2018, month=3, day=1),
    ]
    variable = ["u10m", "v10m", "t2m"]
    ds = xr.Dataset(
        data_vars=dict(
            field1=(
                ["time", "variable", "lat", "lon"],
                np.random.randn(len(time), len(variable), 8, 16),
            ),
            field2=(
                ["time", "variable", "lat", "lon"],
                np.random.randn(len(time), len(variable), 8, 16),
            ),
        ),
        coords={
            "time": time,
            "variable": variable,
        },
    )
    return ds


@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2018, month=1, day=1),
        datetime.datetime(year=2018, month=2, day=1),
    ],
)
@pytest.mark.parametrize("variable", ["u10m", ["u10m", "v10m"]])
def test_data_array_netcdf(foo_data_array, time, variable):
    foo_data_array.to_netcdf("test.nc")
    # Load data source and request data array
    data_source = DataArrayFile("test.nc")
    data = data_source(time, variable)
    # Delete nc file
    pathlib.Path("test.nc").unlink(missing_ok=True)
    # Check consisten
    assert np.all(
        foo_data_array.sel(time=time, variable=variable).values == data.values
    )


@pytest.mark.parametrize("array", ["field1", "field2"])
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2018, month=1, day=1),
        datetime.datetime(year=2018, month=2, day=1),
    ],
)
@pytest.mark.parametrize("variable", ["u10m", ["u10m", "v10m"]])
def test_data_set_netcdf(foo_data_set, array, time, variable):
    foo_data_set.to_netcdf("test.nc")
    # Load data source and request data array
    data_source = DataSetFile("test.nc", array_name=array)
    data = data_source(time, variable)
    # Delete nc file
    pathlib.Path("test.nc").unlink(missing_ok=True)
    # Check consisten
    assert np.all(
        foo_data_set[array].sel(time=time, variable=variable).values == data.values
    )
