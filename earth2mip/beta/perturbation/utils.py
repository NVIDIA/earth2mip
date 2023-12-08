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

from earth2mip.beta.perturbation.base import PertubationMethod


class Perturbation:
    """Applies a perturbation method to an input tensor. If supplied this will filter
    the channel dimension of the input to a subset to apply the specified perturbation.
    Additionally, normalization vectors can be supplied to normalize the data prior to
    applying perturbations.

    Note
    ----
    It is a core design pricinple of Earth-2 MIP to always move data with physical units
    between components. It is very likely users should provide normalization arrays.

    Parameters
    ----------
    method : PertubationMethod
        _description_
    channels : Optional[list[str]], optional
        List of channel id's to apply petrubation on. If None, perturbation will be
        applied to all channels, by default None
    center : Optional[np.ndarray], optional
        Channel center / mean array. If None, no center will be used, by default None
    scale : Optional[np.ndarray], optional
        Channel scale / std array. If None, no scale will be used,, by default None
    """

    def __init__(
        self,
        method: PertubationMethod,
        channels: Optional[list[str]] = None,
        center: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ):
        self.method = method
        self.channels = channels

        if center is None and scale is None:
            center = np.array([0])
            scale = np.array([1])
        elif center is None:
            center = np.zeros_like(scale)
        elif scale is None:
            scale = np.ones_like(center)

        self.center = torch.Tensor(center)
        self.scale = torch.Tensor(scale)

        if self.center.ndim != 1 and self.scale.ndim != 1:
            raise ValueError("Only 1D normalization vectors supported for fields")
        if self.center.shape != self.scale.shape:
            raise ValueError("Center and scale arrays must be the same dimensionality")

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: OrderedDict[str, np.ndarray],
    ) -> tuple[torch.Tensor, OrderedDict[str, np.ndarray]]:
        """Applies perturbation method to input tensor

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : OrderedDict[str, np.ndarray]
            Ordered dict representing coordinate system that discribes the tensor

        Returns
        -------
        tuple[torch.Tensor, OrderedDict[str, np.ndarray]]
            Tensor with applied perturbation, coordinate system
        """
        dims = list(coords.keys())
        # Move channels to first dimension
        x = torch.transpose(x, dims.index("channels"), 0)
        # Filter channels
        if self.channels:
            cindex = torch.IntTensor(
                [coords["channels"].index(i) for i in self.channels]
            ).to(x.device)
            x0 = x[cindex].contiguous()
        else:
            x0 = x

        # Normalize
        center = self.center.to(x.device)
        scale = self.scale.to(x.device)
        for i in range(x.ndim - 1):
            # Padd tail end of tensor dims for broadcasting
            center = center.unsqueeze(-1)
            scale = scale.unsqueeze(-1)

        x0 = (x0 - center) / scale

        # Compute noise
        noise = self.method(x0, coords)
        # Apply noise and unnormalize
        x0 = scale * (x0 + noise) + center

        # Apply channel perturbation
        if self.channels:
            x[cindex] = x0
        else:
            x = x0
        x = torch.transpose(x, 0, dims.index("channels"))

        return x, coords
