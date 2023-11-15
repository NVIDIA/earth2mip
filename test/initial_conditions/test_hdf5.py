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
import pathlib

import h5py
import numpy as np
import pytest

from earth2mip import grid
from earth2mip.initial_conditions import hdf5


def create_hdf5(tmp_path: pathlib.Path, year: int, num_time, grid, channels):
    h5_var_name = "fields"

    data_json = {
        "attrs": {},
        "coords": {"lat": grid.lat, "lon": grid.lon, "channel": channels},
        "dims": ["time", "channel", "lat", "lon"],
        "h5_path": h5_var_name,
        "dhours": 6,
    }

    data_json_path = tmp_path / "data.json"
    data_json_path.write_text(json.dumps(data_json))

    h5_path = tmp_path / "validation" / (str(year) + ".h5")
    h5_path.parent.mkdir()
    with h5py.File(h5_path.as_posix(), mode="w") as f:
        f.create_dataset(
            h5_var_name, shape=(num_time, len(channels), *grid.shape), dtype="<f"
        )

    # create stats/time_means.npy
    time_means = tmp_path / "stats" / "time_means.npy"
    time_means.parent.mkdir()
    time_mean_data = np.zeros([1, len(channels), *grid.shape])
    np.save(time_means.as_posix(), time_mean_data)


@pytest.mark.parametrize("year", [2018, 1980, 2017])
def test__get_path(year):
    root = pathlib.Path(__file__).parent / "mock_data"
    root = root.as_posix()
    dt = datetime.datetime(year, 1, 2)
    path = hdf5._get_path(root, dt)
    assert pathlib.Path(path).name == dt.strftime("%Y.h5")
    assert pathlib.Path(path).exists()


def test__get_path_key_error():

    with pytest.raises(KeyError):
        hdf5._get_path(".", datetime.datetime(2040, 1, 2))


def test_hdf_data_source(tmp_path: pathlib.Path):
    time = datetime.datetime(2018, 1, 1)
    create_hdf5(
        tmp_path,
        time.year,
        10,
        grid=grid.equiangular_lat_lon_grid(2, 2),
        channels=["t850", "t2m"],
    )
    ds = hdf5.DataSource.from_path(tmp_path.as_posix())
    array = ds[time]
    assert array.shape == (2, 2, 2)
    assert isinstance(ds.grid, grid.LatLonGrid)


def test_hdf_data_source_subset_channels(tmp_path: pathlib.Path):
    time = datetime.datetime(2018, 1, 1)
    create_hdf5(
        tmp_path,
        time.year,
        10,
        grid=grid.equiangular_lat_lon_grid(2, 2),
        channels=["t850", "t2m"],
    )
    ds = hdf5.DataSource.from_path(tmp_path.as_posix(), channel_names=["t850"])
    array = ds[time]
    assert array.shape == (1, 2, 2)
    assert isinstance(ds.grid, grid.LatLonGrid)
