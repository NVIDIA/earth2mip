# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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
import numpy as np
import h5py
import os

from einops import rearrange
from earth2mip import schema
import torch_harmonics as th
from earth2mip.networks import Inference
from datetime import datetime
from timeit import default_timer
from typing import Union


class GaussianRandomFieldS2(torch.nn.Module):
    def __init__(
        self,
        nlat,
        alpha=2.0,
        tau=3.0,
        sigma=None,
        radius=1.0,
        grid="equiangular",
        dtype=torch.float32,
    ):
        super().__init__()
        """A mean-zero Gaussian Random Field on the sphere with Matern covariance:
        C = sigma^2 (-Lap + tau^2 I)^(-alpha).

        Lap is the Laplacian on the sphere, I the identity operator,
        and sigma, tau, alpha are scalar parameters.

        Note: C is trace-class on L^2 if and only if alpha > 1.

        Parameters
        ----------
        nlat : int
            Number of latitudinal modes;
            longitudinal modes are 2*nlat.
        alpha : float, default is 2
            Regularity parameter. Larger means smoother.
        tau : float, default is 3
            Lenght-scale parameter. Larger means more scales.
        sigma : float, default is None
            Scale parameter. Larger means bigger.
            If None, sigma = tau**(0.5*(2*alpha - 2.0)).
        radius : float, default is 1
            Radius of the sphere.
        grid : string, default is "equiangular"
            Grid type. Currently supports "equiangular" and
            "legendre-gauss".
        dtype : torch.dtype, default is torch.float32
            Numerical type for the calculations.
        """

        # Number of latitudinal modes.
        self.nlat = nlat

        # Default value of sigma if None is given.
        if sigma is None:
            assert alpha > 1.0, f"Alpha must be greater than one, got {alpha}."
            sigma = tau ** (0.5 * (2 * alpha - 2.0))

        # Inverse SHT
        self.isht = th.InverseRealSHT(
            self.nlat, 2 * self.nlat, grid=grid, norm="backward"
        ).to(dtype=dtype)

        # Square root of the eigenvalues of C.
        sqrt_eig = (
            torch.tensor([j * (j + 1) for j in range(self.nlat)])
            .view(self.nlat, 1)
            .repeat(1, self.nlat + 1)
        )
        sqrt_eig = torch.tril(
            sigma * (((sqrt_eig / radius**2) + tau**2) ** (-alpha / 2.0))
        )
        sqrt_eig[0, 0] = 0.0
        sqrt_eig = sqrt_eig.unsqueeze(0)
        self.register_buffer("sqrt_eig", sqrt_eig)

        # Save mean and var of the standard Gaussian.
        # Need these to re-initialize distribution on a new device.
        mean = torch.tensor([0.0]).to(dtype=dtype)
        var = torch.tensor([1.0]).to(dtype=dtype)
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)

        # Standard normal noise sampler.
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

    def forward(self, N, xi=None):
        """Sample random functions from a spherical GRF.

        Parameters
        ----------
        N : int
            Number of functions to sample.
        xi : torch.Tensor, default is None
            Noise is a complex tensor of size (N, nlat, nlat+1).
            If None, new Gaussian noise is sampled.
            If xi is provided, N is ignored.

        Output
        -------
        u : torch.Tensor
           N random samples from the GRF returned as a
           tensor of size (N, nlat, 2*nlat) on a equiangular grid.
        """
        # Sample Gaussian noise.
        if xi is None:
            xi = self.gaussian_noise.sample(
                torch.Size((N, self.nlat, self.nlat + 1, 2))
            ).squeeze()
            xi = torch.view_as_complex(xi)

        # Karhunen-Loeve expansion.
        u = self.isht(xi * self.sqrt_eig)

        return u

    # Override cuda and to methods so sampler gets initialized with mean
    # and variance on the correct device.
    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

        return self


def generate_noise_correlated(shape, *, reddening, device, noise_amplitude):
    return noise_amplitude * brown_noise(shape, reddening).to(device)


def generate_noise_grf(shape, grid, alpha, sigma, tau):
    sampler = GaussianRandomFieldS2(nlat=720, alpha=alpha, tau=tau, sigma=sigma)
    sample_noise = sampler(shape[0] * shape[1] * shape[2]).reshape(
        shape[0], shape[1], shape[2], 720, 1440
    )
    if grid == schema.Grid.grid_721x1440:
        noise = torch.zeros(shape)
        noise[:, :, :, :-1, :] = sample_noise
        noise[:, :, :, -1:, :] = noise[:, :, :, -2:-1, :]
    else:
        noise = sample_noise
    return noise


def brown_noise(shape, reddening=2):

    noise = torch.normal(torch.zeros(shape), torch.ones(shape))

    x_white = torch.fft.fft2(noise)
    S = (
        torch.abs(torch.fft.fftfreq(noise.shape[-2]).reshape(-1, 1)) ** reddening
        + torch.abs(torch.fft.fftfreq(noise.shape[-1])) ** reddening
    )

    S = torch.where(S == 0, 0, 1 / S)
    S = S / torch.sqrt(torch.mean(S**2))

    x_shaped = x_white * S
    noise_shaped = torch.fft.ifft2(x_shaped).real

    return noise_shaped


def gp_sample(length_scale, num_features=1000, coefficient=1.0):

    x = rearrange(np.mgrid[0:720, 0:1440], "d x y -> (x y) d")
    x = torch.tensor(x).float().cuda()

    omega_shape = (1, num_features, 2)
    omega = torch.normal(
        mean=torch.zeros(omega_shape), std=torch.ones(omega_shape)
    ).cuda()
    omega /= length_scale

    weight_shape = (1, num_features)
    weights = torch.normal(
        mean=torch.zeros(weight_shape), std=torch.ones(weight_shape)
    ).cuda()

    phi = torch.rand((1, num_features, 1)) * (2 * np.pi)
    phi = phi.cuda()

    features = torch.cos(torch.einsum("sfd, nd -> sfn", omega, x) + phi)
    features = (2 / num_features) ** 0.5 * features * coefficient

    functions = torch.einsum("sf, sfn -> sn", weights, features)

    return functions.reshape(1, 720, 1440)


def draw_noise(corr, spreads, length_scales, device):

    z = [gp_sample(l) for l in length_scales]
    z = torch.stack(z, dim=1)

    if spreads is not None:
        sigma = spreads.permute(1, 2, 0)
        A = corr * (sigma[..., None] * sigma[..., None, :])

    else:
        sigma = torch.ones((720, 1440, 34))
        A = corr

    L = torch.linalg.cholesky_ex(A.cpu())[0].to(device)
    z = z[0].permute(1, 2, 0)[..., None]

    return torch.matmul(L, z).permute(3, 2, 0, 1)


def get_skill_spread(ens, obs):
    ensemble_mean = np.mean(ens, axis=0)
    MSE = np.power(obs - ensemble_mean, 2.0)
    RMSE = np.sqrt(MSE)
    spread = np.std(ens, axis=0)
    return RMSE, spread


def load_correlation(grid=schema.Grid.grid_721x1440):
    correlation = torch.load("/lustre/fsw/sw_climate_fno/ensemble_init_stats/corr.pth")
    if grid == schema.Grid.grid_721x1440:
        correlation = torch.cat((correlation, correlation[-1, :, :, :].unsqueeze(0)), 0)
    return correlation


# ~~~~~~~~~~~~~~SPHERICAL GRF~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Complex outer product in the channel dimension
# a : (batch_size, channels_a, s1, s2)
# b : (batch_size, channels_b, s1, s2)
# out : (batch_size, channels_a, channels_b, s1, s2)
def complex_outer(a, b):
    return torch.einsum("bixy, bjxy -> bijxy", a, torch.conj(b))


def learn_spherical_covariance(
    path,
    files=["2016.h5"],
    s1=721,
    s2=1440,
    channels=73,
    grid="equiangular",
    eps=1e-5,
    device=None,
    verbose=True,
):

    # Initialize SHT for spherical data on (s1, s2) grid
    sht = th.RealSHT(s1, s2, grid=grid).to(dtype=torch.float32).to(device=device)

    # Let Z denote a complex multi-dimensional Gaussian vector
    # We consider s1*floor(s2/2 +1) such i.i.d. Zs each of dimension channels
    s2_out = int(s2 / 2.0 + 1)

    # E[Z]
    mean = torch.zeros((1, channels, s1, s2_out), device=device, dtype=torch.cfloat)
    # E[Z * Z^H]
    squared_mean = torch.zeros(
        (1, channels, channels, s1, s2_out), device=device, dtype=torch.cfloat
    )

    # Total number of data points
    total_n = 0

    era5_mean = torch.from_numpy(np.load(path + "/stats/global_means.npy")).to(
        dtype=torch.float32
    )[:, 0:channels, ...]
    era5_std = torch.from_numpy(np.load(path + "/stats/global_stds.npy")).to(
        dtype=torch.float32
    )[:, 0:channels, ...]

    # Monte-Carlo estimate of E[Z] and E[Z * Z^H] on the fly
    for file_name in files:
        f = h5py.File(path + file_name, "r")
        n_examp = f["fields"].shape[0]

        print(f"Total examples: {n_examp}")
        for j in range(n_examp):
            t1 = default_timer()

            batch = (
                torch.from_numpy(f["fields"][j, 0:channels, ...])
                .to(dtype=torch.float32)
                .unsqueeze(0)
            )
            batch -= era5_mean
            batch /= era5_std

            # Items in the batch
            current_n = batch.size(0)

            if device is not None:
                batch = batch.to(device)

            # Compute SHT of data
            batch = sht(batch)

            # Update running estimates of E[Z] and E[Z * Z^H]
            const = 1.0 / (total_n + current_n)
            mean = const * (total_n * mean + torch.sum(batch, dim=0, keepdim=True))
            squared_mean = const * (
                total_n * squared_mean
                + torch.sum(complex_outer(batch, batch), dim=0, keepdim=True)
            )

            total_n += current_n

            if verbose:
                print(f"Processed examples: {total_n}, Time: {default_timer() - t1}")

    # Covariance = E[Z * Z^H] - E[Z] * E[Z]^H
    covar = squared_mean - complex_outer(mean, mean)

    # Covariances to (s1*s2_out, channels, channels)
    covar.squeeze_(0)
    covar = covar.permute(2, 3, 0, 1).view(-1, channels, channels)

    not_decomposed = True

    while not_decomposed:
        try:
            print(f"Trying Cholesky with nugget: {eps}")

            # Nugget for numerical stability
            identity = eps * torch.eye(
                channels, channels, device=device, dtype=torch.cfloat
            ).unsqueeze(0)
            # Batch-wise Cholesky of the covariances
            sqrt_covar = torch.linalg.cholesky(covar + identity)

            # Success
            not_decomposed = False

        except BaseException:
            # If fails, increase eps
            eps *= 10

            if eps >= 10.0:
                print("Failed Cholesky. Exiting...")
                return None, None

    # Reshape to (channels, channels s1, s2_out)
    sqrt_covar = sqrt_covar.view(s1, s2_out, channels, channels).permute(2, 3, 0, 1)

    return mean.squeeze_(0), sqrt_covar


# Sample from a channel correlated spherical GRF
# mean : (channels, s1, s2)
# sqrt_var : (channels, channels, s1, s2)
class CorrelatedGRFS2(torch.nn.Module):
    def __init__(
        self,
        mean,
        sqrt_covar,
        s1=721,
        s2=1440,
        channels=73,
        alpha=0.3,
        grid="equiangular",
    ):

        super().__init__()

        # Remember parameters
        self.channels = channels
        self.s1 = s1
        self.s2 = s2
        self.alpha = alpha
        self.s2_out = int(s2 / 2.0 + 1)

        # Initialize inverse SHT
        self.isht = th.InverseRealSHT(s1, s2, grid=grid).to(dtype=torch.float32)

        # Store means and square roots of covariances
        self.register_buffer("mean", mean.unsqueeze(0))
        self.register_buffer("sqrt_covar", sqrt_covar)

        # Save mean and standard deviation of the standard complex Gaussian
        # Need these to re-initialize distribution on a new device
        self.register_buffer("std_mean", torch.tensor([0.0]).to(dtype=torch.float32))
        self.register_buffer("std_var", torch.tensor([0.5]).to(dtype=torch.float32))

        # Register smoothing mask
        if self.alpha > 0.0:
            sqrt_eig = (
                torch.tensor([j * (j + 1) for j in range(self.s1)])
                .view(self.s1, 1)
                .repeat(1, self.s2_out)
            )
            sqrt_eig = torch.tril((sqrt_eig) ** (-self.alpha))
            sqrt_eig[0, 0] = 1.0
            sqrt_eig[-1, -1] = 0.0
            sqrt_eig = sqrt_eig.unsqueeze(0).unsqueeze(0)
            self.register_buffer("smooth_mask", sqrt_eig)

        # Mask to remove unneccecery modes
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.s1, self.s2_out)).unsqueeze(0).unsqueeze(0),
        )

        # Standard normal noise sampler
        self.gaussian_noise = torch.distributions.normal.Normal(
            self.std_mean, self.std_var
        )

    # Sample the GRF
    # sample_size is integer number of desired samples
    # out : (sample_size, channels, s1, s2)
    def forward(self, sample_size):

        # Sample from the standard complex Guassian
        sample = self.gaussian_noise.sample(
            torch.Size((sample_size, self.channels, self.s1, self.s2_out, 2))
        ).squeeze(5)
        sample = torch.view_as_complex(sample)

        # Batch-wise matrix multiply by square root of the covariance and add mean
        sample = torch.einsum("abcd, ebcd -> eacd", self.sqrt_covar, sample) + self.mean
        sample = sample * self.mask

        if self.alpha > 0.0:
            # Energy of unsmoothed sample
            original_energy = torch.sqrt(
                torch.sum(torch.abs(sample) ** 2, dim=[2, 3], keepdim=True)
            )

            # Apply smoothing mask
            sample = sample * self.smooth_mask

            # Energy of smoothed sample
            smooth_energy = torch.sqrt(
                torch.sum(torch.abs(sample) ** 2, dim=[2, 3], keepdim=True)
            )

            # Preserve energy of the original
            sample = (original_energy / smooth_energy) * sample

        # Turn sample to physical space a.k.a. compute Karhunen-Loeve expansion
        sample = self.isht(sample)

        return sample

    # Override "cuda" and "to" methods so sampler gets initialized with mean
    # and standard deviation on the correct device.
    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(
            self.std_mean, self.std_var
        )

        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(
            self.std_mean, self.std_var
        )

        return self


def load_spherical_mean_covar():
    data_path = os.environ.get("DATA")
    if data_path is None:
        data_path = "/lustre/fsw/sw_climate_fno"

    try:
        mean = torch.load(data_path + "/era5_mean_ch73_sht.pt")
        sqrt_covar = torch.load(data_path + "/era5_sqrtcovar_ch73_sht.pt")
    except BaseException:
        mean = None
        sqrt_covar = None
        print("Warning: unable to load data mean and covaraince for initialization.")

    return mean, sqrt_covar


def generate_correlated_spherical_grf(
    shape, grid, mean, sqrt_covar, alpha=None, sigma=None, tau=None
):

    if mean is None or sqrt_covar is None:
        return generate_noise_grf(shape, grid, alpha, sigma, tau)

    sampler = CorrelatedGRFS2(mean, sqrt_covar, channels=shape[2])

    sample_noise = sampler(shape[0] * shape[1]).reshape(
        shape[0], shape[1], shape[2], 721, 1440
    )

    if sigma is not None:
        sample_noise = sigma * sample_noise

    if grid != schema.Grid.grid_721x1440:
        sample_noise = sample_noise[:, :, :, 0:-1, :]  # 720 x 1440

    return sample_noise


def generate_bred_vector(
    x: torch.Tensor,
    model: Inference,
    noise_amplitude: float = 0.15,
    time: Union[datetime, None] = None,
    integration_steps: int = 40,
    inflate=False,
):

    # Assume x has shape [ENSEMBLE, TIME, CHANNEL, LAT, LON]
    x0 = x[:1]

    # Get control forecast
    for data in model.run_steps(x0, n=1, normalize=False, time=time):
        xd = data

    # Unsqueeze if time has been collapsed.
    if xd.ndim != x0.ndim:
        xd = xd.unsqueeze(1)

    dx = noise_amplitude * torch.randn(x.shape, device=x.device, dtype=x.dtype)
    for _ in range(integration_steps):
        x1 = x + dx
        for data in model.run_steps(x1, n=1, normalize=False, time=time):
            x2 = data

        # Unsqueeze if time has been collapsed.
        if x2.ndim != x1.ndim:
            x2 = x2.unsqueeze(1)
        dx = x2 - xd

        if inflate:
            dx += noise_amplitude * (dx - dx.mean(dim=0))

    gamma = torch.norm(x) / torch.norm(x + dx)
    return noise_amplitude * dx * gamma
