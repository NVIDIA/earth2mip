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
import xarray as xr


class BrownNoise:
    """Applied 2D brown noise to the lat / lon dimensions of the input tensor.

    Parameters
    ----------
    reddening : int, optional
        Reddening in Fourier space, by default 2
    noise_amplitude : float, optional
        Noise amplitude, by default 0.05
    """

    def __init__(self, reddening: int = 2, noise_amplitude: float = 0.05):
        self.reddening = reddening
        self.noise_amplitude = noise_amplitude

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
            Input tensor
        xrc : xr.Coordinates
            Xarray coordinate system, must contain coordinates "lat" and "lon"

        Returns
        -------
        torch.Tensor
            Output tensor with applied perturbation
        """
        shape = x.shape
        dims = xrc.dims
        # Can be generalized if needed to have coords a constructor parameter
        if "lat" not in dims or "lon" not in dims:
            raise ValueError("Input tensor coords needs to contian lat and lon dims")

        ilat = xrc.dims.index("lat")
        ilon = xrc.dims.index("lon")

        # Move lat, lon to last coordinates for noise generation
        shape0 = list(shape)
        for i in sorted([ilat, ilon], reverse=True):
            del shape0[i]
        shape0 = shape0 + [shape[ilat], shape[ilon]]

        noise = self._generate_noise_correlated(tuple(shape0), device=x.device)
        print(noise.shape)
        noise = torch.moveaxis(noise, (-2, -1), (ilat, ilon))

        return x + self.noise_amplitude * noise

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


if __name__ == "__main__":

    noise = BrownNoise()

    import numpy as np

    input_tensor = torch.randn(3, 16, 32)
    # xrc = xr.Coordinates(coords={
    #         "time": np.linspace(0, 1, input_tensor.shape[0]).tolist(),
    #         "lat": np.linspace(-90, 90, input_tensor.shape[1]).tolist(),
    #         "lon": np.linspace(0, 360, input_tensor.shape[1]).tolist()
    #     }, indexes={})

    xrc = xr.DataArray(
        dims=["time", "lon", "lat"],
        coords={
            "time": np.linspace(0, 1, input_tensor.shape[0]).tolist(),
            "lat": np.linspace(-90, 90, input_tensor.shape[1]).tolist(),
            "lon": np.linspace(0, 360, input_tensor.shape[1]).tolist(),
        },
    ).coords

    noise(input_tensor, xrc)
