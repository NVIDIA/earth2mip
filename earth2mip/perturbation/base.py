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

from typing import Protocol, runtime_checkable

import torch
import xarray as xr


@runtime_checkable
class PertubationMethod(Protocol):
    """Perturbation interface."""

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        xrc: xr.Coordinates,
    ) -> torch.Tensor:
        """Apply perturbation method

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to apply perturbation on
        xrc : xr.Coordinates
            Xarray coordinate system that discribes the tensor

        Returns
        -------
        torch.Tensor
            Ouput tensor, on the same coordinate system, with applied perturbation
        """
        pass
