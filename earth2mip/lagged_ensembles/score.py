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
import logging

import torch

import earth2mip.grid
from earth2mip.crps import crps_from_empirical_cdf

logger = logging.getLogger(__name__)


def weighted_average(x, w, dim):
    return torch.mean(x * w, dim) / torch.mean(w, dim)


def area_average(grid: earth2mip.grid.LatLonGrid, x: torch.Tensor):
    lat = torch.tensor(grid.lat, device=x.device)[:, None]
    cos_lat = torch.cos(torch.deg2rad(lat))
    return weighted_average(x, cos_lat, dim=[-2, -1])


def score(
    grid: earth2mip.grid.LatLonGrid,
    ensemble: dict[int, torch.Tensor],
    obs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Set of standardized scores for lagged ensembles

    Includes:
    - crps
    - mse of ensemble mean
    - variance about ensemble mean
    - MSE of the first ensemble member

    Args:
        ensemble: mapping of lag to (c, ...)
        obs: (c, ...)

    Returns:
        dict of str to (channel,) shaped metrics tensors.

    """
    # need to run this after since pandas.Index doesn't support cupy
    out = {}
    ens = torch.stack(list(ensemble.values()), dim=0)
    ensemble_dim = 0

    # compute all the metrics
    num = crps_from_empirical_cdf(obs, ens)
    out["crps"] = area_average(grid, num)

    num = (ens.mean(ensemble_dim) - obs) ** 2
    out["MSE_mean"] = area_average(grid, num)

    num = ens.var(ensemble_dim, unbiased=True)
    out["variance"] = area_average(grid, num)

    if 0 in ensemble:
        i = list(ensemble.keys()).index(0)
        num = (ens[i] - obs) ** 2
        out["MSE_det"] = area_average(grid, num)

    return {k: v.ravel() for k, v in out.items()}
