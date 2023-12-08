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
import torch
import xarray as xr

from earth2mip.beta.data.utils import prep_data_array


@pytest.fixture
def foo_data_array():
    time0 = datetime.datetime.now()
    return xr.DataArray(
        data=np.random.rand(8, 16, 32),
        dims=["one", "two", "three"],
        coords={
            "one": [time0 + i * datetime.timedelta(hours=6) for i in range(8)],
            "two": [f"{i}" for i in range(16)],
            "three": np.linspace(0, 1, 32),
        },
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dims", [["one", "two", "three"], ["three", "one", "two"]])
def test_prep_dataarray(foo_data_array, dims, device):

    data_array = foo_data_array.transpose(*dims)
    out, outc = prep_data_array(data_array, device)

    assert str(out.device) == device
    assert list(outc.keys()) == list(data_array.dims)
    for key in outc.keys():
        assert (outc[key] == np.array(data_array.coords[key])).all()
    assert out.shape == data_array.data.shape
