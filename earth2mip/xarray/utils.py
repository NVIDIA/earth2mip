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


def concat_dict(d, key_names=(), concat_dim="key"):
    """concat a dictionary of xarray objects"""
    arrays = []
    for key, arr in d.items():
        coords = {}
        for name, value in zip(key_names, key):
            coords[name] = xarray.Variable([concat_dim], [value])
        arr = arr.expand_dims(concat_dim)
        arr = arr.assign_coords(coords)
        arrays.append(arr)
    return xarray.concat(arrays, dim=concat_dim)
