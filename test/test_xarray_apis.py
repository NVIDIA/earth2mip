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

"""
xarray is confusing so let's test it to gain understanding
"""
import numpy as np
import pytest
import xarray


def test_xarray_var_reference():
    ds = xarray.DataArray(np.ones((10, 10)), dims=("x", "y"))
    ds["x"] = np.arange(10)
    ds["y"] = np.arange(10)
    datasets = ds.to_dataset(name="wind")
    datasets["var"] = datasets["wind"]
    assert isinstance(datasets["var"], xarray.DataArray)


def test_xarray_loop():
    ds = xarray.DataArray(np.ones((10, 10)), dims=("x", "y"))
    ds["x"] = np.arange(10)
    ds["y"] = np.arange(10)
    datasets = ds.to_dataset(name="wind")

    assert list(datasets) == ["wind"]
    assert set(datasets.variables) == {"x", "y", "wind"}


def test_xarray_var():
    ds = xarray.DataArray(np.ones((10, 10)), dims=("x", "y"))
    ds["x"] = np.arange(10)
    ds["y"] = np.arange(10)
    datasets = ds.to_dataset(name="wind")

    with pytest.raises(AttributeError):
        datasets.variables["wind"].weighted
