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

import xarray
import numpy as np
from earth2mip.xarray.utils import to_cupy, concat_dict

try:
    import cupy
except ImportError:
    cupy = None

import pytest


@pytest.fixture()
def has_gpu():
    try:
        cupy.cuda.runtime.getDevice()
    except Exception:
        pytest.skip("No GPU.")


@pytest.mark.parametrize("chunked", [True, False])
def test_to_cupy_weighted(has_gpu, chunked):
    nlat, nlon = 10, 20
    lat = xarray.Variable(["lat"], np.linspace(90, -90, nlat))
    arr = xarray.DataArray(
        np.ones([nlat, nlon]), dims=["lat", "lon"], coords={"lat": lat}
    )
    arr_cp = to_cupy(arr)
    if chunked:
        arr_cp = arr_cp.chunk({"lat": 2})

    arr_cp.load().data.device


def test_concat_dict():
    # Create sample input dictionary of xarray objects
    data1 = xarray.DataArray([1, 2, 3], dims="dim1")
    data2 = xarray.DataArray([4, 5, 6], dims="dim1")
    data3 = xarray.DataArray([7, 8, 9], dims="dim1")

    input_dict = {
        ("coord1_val1",): data1,
        ("coord1_val2",): data2,
        ("coord1_val3",): data3,
    }

    # Call the function
    result = concat_dict(input_dict, key_names=("coord1",), concat_dim="key")

    coord = xarray.Variable(["key"], [k[0] for k in input_dict])
    expected_values = xarray.DataArray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dims=("key", "dim1"),
        coords={"coord1": coord},
    )
    xarray.testing.assert_equal(result, expected_values)
