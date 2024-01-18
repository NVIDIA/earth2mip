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

import numpy as np
import torch

from earth2mip.beta.utils import handshake_dim


class Brown:
    """Lat/Lon 2D brown noise

    Parameters
    ----------
    noise_amplitude : float, optional
        Noise amplitude, by default 0.05
    reddening : int, optional
        Reddening in Fourier space, by default 2
    """

    def __init__(self, noise_amplitude: float = 0.05, reddening: int = 2):
        self.reddening = reddening
        self.noise_amplitude = noise_amplitude

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: OrderedDict[str, np.ndarray],
    ) -> torch.Tensor:
        """Apply perturbation method

        Parameters
        ----------
        x : torch.Tensor
            Input tensor intended to apply perturbation on
        coords : OrderedDict[str, np.ndarray]
            Ordered dict representing coordinate system that discribes the tensor, must
            contain "lat" and "lon" coordinates in last two dims

        Returns
        -------
        torch.Tensor
            Perturbation noise tensor
        """
        shape = x.shape
        # Check the required dimensions are present
        handshake_dim(coords, required_dim="lat", required_index=-2)
        handshake_dim(coords, required_dim="lon", required_index=-1)

        noise = self._generate_noise_correlated(tuple(shape), device=x.device)

        return self.noise_amplitude * noise

    def _generate_noise_correlated(
        self, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        """Utility class for producing brown noise."""
        noise = torch.randn(*shape, device=device)
        x_white = torch.fft.rfft2(noise)
        S = (
            torch.abs(torch.fft.fftfreq(shape[-2], device=device).reshape(-1, 1))
            ** self.reddening
            + torch.fft.rfftfreq(shape[-1], device=device) ** self.reddening
        )
        S = 1 / S
        S[..., 0, 0] = 0
        S = S / torch.sqrt(torch.mean(S**2))

        x_shaped = x_white * S
        noise_shaped = torch.fft.irfft2(x_shaped, s=shape[-2:])
        return noise_shaped
