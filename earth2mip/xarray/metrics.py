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

# TODO consolidate with fcn_mip/score_ensemble_outputs.py
from contextlib import contextmanager
from typing import List, Mapping

import numpy as np
import xarray
import xskillscore


@contextmanager
def properscoring_with_cupy():
    """A context manager that makes proper-scoring compatible with cupy arrays

    Not thread safe.
    """
    import cupy
    import properscoring

    # using cupy is 3x faster
    properscoring._crps.np = cupy
    old_score = properscoring._crps._crps_ensemble_core

    properscoring._crps._crps_ensemble_core = (
        properscoring._crps._crps_ensemble_vectorized
    )
    yield
    properscoring._crps.np = np
    properscoring._crps._crps_ensemble_core = old_score


def global_average(x, lat):
    cos_lat = np.cos(np.deg2rad(lat))
    return x.weighted(cos_lat).mean(dim=["lat", "lon"])


def score_ensemble(
    ensemble: np.ndarray, obs: np.ndarray, lat: np.ndarray, ensemble_keys: List[int]
) -> Mapping[str, xarray.DataArray]:
    """Set of standardized scores for ensembles

    Includes:
    - crps
    - mse of ensemble mean
    - variance about ensemble mean
    - MSE of the first ensemble member

    """
    out = {}
    num = (ensemble.mean("ensemble") - obs) ** 2
    out["MSE_mean"] = global_average(num.load(), lat)

    num = ensemble.var("ensemble", ddof=1)
    out["variance"] = global_average(num.load(), lat)

    if 0 in ensemble_keys:
        num = (ensemble.isel(ensemble=ensemble_keys.index(0)) - obs) ** 2
        out["MSE_det"] = global_average(num.load(), lat)

    crps = xskillscore.crps_ensemble(
        obs,
        ensemble,
        issorted=False,
        member_dim="ensemble",
        dim=(),
        keep_attrs=True,
    )
    out["crps"] = global_average(crps.load(), lat)
    return out
