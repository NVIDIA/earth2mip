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

from earth2mip.perturbation.base import PertubationMethod


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

        if not center:
            center = np.array([0])
        if not scale:
            scale = np.array([1])

        self.center = torch.Tensor(center)
        self.scale = torch.Tensor(scale)

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
        # Filter channels
        if self.channels:
            cindex = torch.IntTensor([dims.index(dim) for dim in self.channels])
            x0 = torch.index_select(x, dim=dims.index("channels"), index=cindex)
        else:
            x0 = x

        # Normalize
        center = self.center.to(x.device)
        scale = self.scale.to(x.device)
        for i in range(len(dims[dims.index("channels") + center.ndim :])):
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
            for index in cindex:
                torch.select(x, dims.index("channels"), x0)
        else:
            x = x0

        return x, coords
