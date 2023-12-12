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

from collections import OrderedDict

import pytest

from earth2mip.beta.utils import handshake_dim


@pytest.mark.parametrize(
    "coords",
    [
        OrderedDict([("batch", []), ("variable", []), ("lat", []), ("lon", [])]),
        OrderedDict([("time", []), ("lat", []), ("lon", [])]),
    ],
)
def test_handshake_dim(coords):
    # Check dims no index
    for dim in list(coords.keys()):
        handshake_dim(coords, dim)
    # Check dims with index
    for i, dim in enumerate(list(coords.keys())):
        handshake_dim(coords, dim, i)
    # Check dims with reverse index
    for i, dim in enumerate(list(coords.keys())[::-1]):
        handshake_dim(coords, dim, -(i + 1))


@pytest.mark.parametrize(
    "coords",
    [
        OrderedDict([("a", []), ("b", []), ("lat", []), ("lon", [])]),
        OrderedDict([("lat", []), ("lon", [])]),
    ],
)
def test_handshake_dim_failure(coords):

    with pytest.raises(KeyError):
        handshake_dim(coords, "fake_dim")

    with pytest.raises(ValueError):
        handshake_dim(coords, "lat", -1)

    with pytest.raises(ValueError):
        handshake_dim(coords, "lat", 5)
