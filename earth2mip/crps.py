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
import torch


def crps_from_empirical_cdf(
    truth: torch.Tensor, ensemble: torch.Tensor
) -> torch.Tensor:
    """Compute the exact CRPS using the CDF method

    Uses this formula
        # int [F(x) - 1(x-y)]^2 dx

    where F is the emperical CDF and 1(x-y) = 1 if x > y.

    This method is more memory efficient than the kernel method, and uses O(n
    log n) compute instead of O(n^2), where n is the number of ensemble members.

    Args:
        truth: (...) tensor of observations
        ensemble: (N, ...) tensor of ensemble members

    Returns:
        (...,) tensor of CRPS scores


    """
    # TODO upstream this to modulus core
    y = truth
    n = ensemble.shape[0]
    ensemble, _ = torch.sort(ensemble, dim=0)
    ans = 0

    # dx [F(x) - H(x-y)]^2 = dx [0 - 1]^2 = dx
    val = ensemble[0] - y
    ans += torch.where(val > 0, val, 0.0)

    for i in range(n - 1):
        x0 = ensemble[i]
        x1 = ensemble[i + 1]

        cdf = (i + 1) / n

        # a. case y < x0
        val = (x1 - x0) * (cdf - 1) ** 2
        mask = y < x0
        ans += torch.where(mask, val, 0.0)

        # b. case x0 <= y <= x1
        val = (y - x0) * cdf**2 + (x1 - y) * (cdf - 1) ** 2
        mask = (y >= x0) & (y <= x1)
        ans += torch.where(mask, val, 0.0)

        # c. case x1 < t
        mask = y > x1
        val = (x1 - x0) * cdf**2
        ans += torch.where(mask, val, 0.0)

    # dx [F(x) - H(x-y)]^2 = dx [1 - 0]^2 = dx
    val = y - ensemble[-1]
    ans += torch.where(val > 0, val, 0.0)
    return ans
