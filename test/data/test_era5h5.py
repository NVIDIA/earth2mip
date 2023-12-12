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
import json
import os
import pathlib

import h5py
import numpy as np
import pytest

from earth2mip.beta.data import ERA5H5


@pytest.fixture(scope="session")
def test_h5_dataset_1deg(tmp_path_factory) -> str:
    tmp_path = tmp_path_factory.mktemp("data")
    num_time = 4
    lat = np.linspace(90, 90, 180)
    lon = np.linspace(0, 359, 360)
    variable = ["u10m", "v10m", "t2m", "msl"]

    data_json = {
        "coords": {"lat": lat.tolist(), "lon": lon.tolist(), "variable": variable},
        "dt_hours": 6,
    }

    data_json_path = os.path.join(tmp_path, "data.json")
    with open(data_json_path, "w") as f:
        json.dump(data_json, f)

    pathlib.Path(os.path.join(tmp_path, "validation")).mkdir(
        parents=True, exist_ok=True
    )
    h5_path = pathlib.Path(tmp_path, "validation", "2000.h5")
    h5_path.parent.mkdir(exist_ok=True)
    with h5py.File(h5_path.as_posix(), mode="w") as f:
        f.create_dataset(
            "fields", shape=(num_time, len(variable), len(lat), len(lon)), dtype="<f"
        )

    h5_path = pathlib.Path(tmp_path, "2001.h5")
    h5_path.parent.mkdir(exist_ok=True)
    with h5py.File(h5_path.as_posix(), mode="w") as f:
        f.create_dataset(
            "fields", shape=(num_time, len(variable), len(lat), len(lon)), dtype="<f"
        )

    return tmp_path.as_posix()


@pytest.mark.slow
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2000, month=1, day=1),
        [
            datetime.datetime(year=2000, month=1, day=1, hour=6),
            datetime.datetime(year=2001, month=1, day=1, hour=12),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["u10m", "v10m", "msl"]])
def test_era5h5_local(time, variable, test_h5_dataset_1deg):

    ds = ERA5H5(test_h5_dataset_1deg)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 180
    assert shape[3] == 360
    assert not np.isnan(data.values).any()


# TODO: (also needed to check cache implementation)
# def test_era5h5_s3(time, variable, test_h5_dataset_1deg):
#     pass


@pytest.mark.slow
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=2002, month=1, day=1),
        datetime.datetime(year=2001, month=1, day=1, hour=1),
    ],
)
@pytest.mark.parametrize("variable", ["t2m", ["u10m", "v10m"]])
def test_era5h5_failures_value(time, variable, test_h5_dataset_1deg):
    # These errors we should catch prior to even loading a H5 file
    ds = ERA5H5(test_h5_dataset_1deg)
    with pytest.raises(ValueError):
        ds(time, variable)


@pytest.mark.slow
@pytest.mark.parametrize(
    "time,variable",
    [
        [datetime.datetime(year=2001, month=1, day=1, hour=6), "fake"],
        [datetime.datetime(year=2001, month=2, day=1, hour=6), "t2m"],
    ],
)
def test_era5h5_failures_key(time, variable, test_h5_dataset_1deg):

    ds = ERA5H5(test_h5_dataset_1deg)
    with pytest.raises(KeyError):
        ds(time, variable)
