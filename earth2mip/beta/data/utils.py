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
from typing import Optional

import numpy as np
import torch
import xarray as xr


def prep_data_array(
    da: xr.DataArray, device: Optional[torch.device] = "cpu"
) -> tuple[torch.Tensor, OrderedDict[str, np.ndarray]]:
    """Prepares a data array from a data source for inference workflows by converting
    the data array to a torch tensor and the coordinate system to an OrderedDict.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    device : torch.device, optional
        Torch devive to load data tensor to, by default "cpu"

    Returns
    -------
    tuple[torch.Tensor, OrderedDict[str, np.ndarray]]
        Tuple containing output tensor and coordinate OrderedDict
    """
    out = torch.Tensor(da.values).to(device)

    out_coords = OrderedDict()
    for dim in da.coords.dims:
        out_coords[dim] = np.array(da.coords[dim])

    return out, out_coords
