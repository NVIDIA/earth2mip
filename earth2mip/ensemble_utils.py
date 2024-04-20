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

from datetime import datetime, timedelta
from typing import Union
import xarray as xr

import torch
import torch_harmonics as th
import numpy as np

from earth2mip.time_loop import TimeLoop
from earth2mip import initial_conditions


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
            assert alpha > 1.0, f"Alpha must be greater than one, got {alpha}."  # noqa
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


def generate_noise_grf(shape, grid, alpha, sigma, tau, device=None):
    sampler = GaussianRandomFieldS2(nlat=720, alpha=alpha, tau=tau, sigma=sigma).to(
        device
    )
    sample_noise = sampler(shape[0] * shape[1] * shape[2]).reshape(
        shape[0], shape[1], shape[2], 720, 1440
    )
    if grid.shape == (721, 1440):
        noise = torch.zeros(shape).to(device)
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

def generate_bred_vector_timeevolve(
    x: torch.Tensor,
    model: TimeLoop,
    noise_amplitude: torch.Tensor,
    weather_event, 
    time: Union[datetime, None] = None,
    integration_steps: int = 3,
    inflate=False,
) -> torch.Tensor:
    # Assume x has shape [ENSEMBLE, TIME, CHANNEL, LAT, LON]

    if isinstance(noise_amplitude, float):
        noise_amplitude = torch.tensor([noise_amplitude]).to(x.device) 
    assert (noise_amplitude.shape[0] == x.shape[2]) or (  # noqa
        torch.numel(noise_amplitude) == 1
    )

    optimization_target = _load_optimal_targets('sfno_linear_73chq_sc3_layers8_edim384_wstgl2', 48, model.channel_names).to(x.device).squeeze().to(torch.float32) * 0.35
    #optimization_target = 0.2 * torch.from_numpy(np.load("/pscratch/sd/p/pharring/74var-6hourly/staging/stats_fcndev/time_diff_stds.npy")).to(x.device).squeeze().to(torch.float32)
    #optimization_target = graphcast_weights(model.channel_names, model.scale, "deterministic rmse").to(x.device).squeeze().to(torch.float32)
    #time_means = torch.from_numpy(np.load("/pscratch/sd/p/pharring/74var-6hourly/staging/stats_fcndev/time_means.npy")).to(x.device).to(torch.float32)
    #optimization_target = calculate_global_geometric_mean(time_means.squeeze()) * 0.05
    sampler = CorrelatedSphericalField(720, 500. * 1000, 48.0, 1.0, N=74).to(
        x.device
    )
    correlated_random_noise = sampler()
    #correlated_random_noise = torch.cat([correlated_random_noise] * 74, dim=1)
    original_pert = correlated_random_noise.clone().detach()
    dx = optimization_target[:, None, None] * correlated_random_noise
    #dx = model.scale * correlated_random_noise * 0.05
    #dx = (correlated_random_noise * 0.05) * x[0]
    #channel_idx = set(range(len(model.channel_names))) - set([model.channel_names.index('z500')])
    channel_idx = list(range(len(model.channel_names)))
    channel_idx.remove(model.channel_names.index('z500'))
    #channel_idx = []
    #for idx, channel in enumerate(model.channel_names):
    #    #if channel in ['q1000','q850']:
    #    if channel in ["z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500"]: 
    #        #these channels will be perturbed
    #        pass
    #    else:
    #        #these channels will not be perturbed
    #        channel_idx.append(idx)
    dx[ :, list(channel_idx), : 721, : 1440] = 0

    data_source = initial_conditions.get_data_source(
        model.in_channel_names,
        initial_condition_source=weather_event.properties.initial_condition_source,
        netcdf=weather_event.properties.netcdf,
    )

    date_obj = weather_event.properties.start_time
    #lag = np.random.choice(np.arange(0,49,6)).astype(float)
    #lagged_time = date_obj - timedelta(hours=lag)
    #lagged_ic = initial_conditions.get_initial_condition_for_model(model, data_source, lagged_time)
    #exaggerate_factor = calculate_global_rms(lagged_ic - x) / optimization_target[None, None]
    #dx = dx / exaggerate_factor[:, :, :, None, None]
    #return (lagged_ic - x) / model.scale / 50

    for step in list(range(integration_steps))[::-1]:
        x0 = initial_conditions.get_initial_condition_for_model(model, data_source, date_obj - (timedelta(hours=6*step)))

        # Get control forecast
        for k, (_, data, _) in enumerate(model(time, x0, model_perturb=False)):
            xd = data
            if k == 1:
                break

        # Unsqueeze if time has been collapsed.
        if xd.ndim != x0.ndim:
            xd = xd.unsqueeze(1)

        x1 = x0 + dx
        for k, (_, data, _) in enumerate(model(time, x1, model_perturb=False)):
            x2 = data
            if k == 1:
                break

        # Unsqueeze if time has been collapsed.
        if x2.ndim != x1.ndim:
            x2 = x2.unsqueeze(1)
        dx = x2 - xd

        if inflate:
            dx += noise_amplitude * (dx - dx.mean(dim=0)) * model.scale

        #exaggerate_factor = calculate_global_rms(dx) / optimization_target[None, None]
        #dx = dx / exaggerate_factor[:, :, :, None, None]

        #exaggerate_factor = calculate_global_geometric_mean(dx) / optimization_target[None, None]
        #dx = dx / exaggerate_factor[:, :, :, None, None]

        exaggerate_factor = hemispheric_rms(dx) / optimization_target[None, None, :, None]
        dx = dx / exaggerate_factor[:, :, :, : , None]

        #dx = set_50hPa_to_0(dx, model.channel_names)
        
        #sampler = CorrelatedSphericalField(720, 2000. * 1000, 48.0, 0.005, N=1).to(
        #    x.device
        #)
        #correlated_random_noise = sampler()
        #correlated_random_noise = torch.cat([correlated_random_noise] * 74, dim=1)
        #channels_to_exclude = []
        #for idx, channel in enumerate(model.channel_names):
        #    #if channel in ['z500']: 
        #    if channel in ["z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500"]: 
        #        #these channels will be perturbed
        #        pass
        #    else:
        #        #these channels will not be perturbed
        #        channels_to_exclude.append(idx)
        #correlated_random_noise[:, torch.Tensor(np.asarray(channels_to_exclude)).to(torch.int), : 721, : 1440] = 0
        #dx = dx + (x2 * correlated_random_noise)
        #dx = dx + (optimization_target[:, None, None] * correlated_random_noise)
    
    #dx = dx + original_pert
    #optimization_target = optimization_target[None, None]
    #exaggerate_factor = calculate_global_rms(dx) / optimization_target
    #dx = dx / exaggerate_factor[:, :, :, None, None]

    #optimization_target = optimization_target[None, None, :, None]
    #exaggerate_factor = hemispheric_rms(dx) / optimization_target
    #dx = dx / exaggerate_factor[:, :, :, : , None]

    return dx / model.scale

def set_50hPa_to_0(dx, channel_names):
    t50_idx = channel_names.index('t50')
    u50_idx = channel_names.index('u50')
    v50_idx = channel_names.index('v50')
    z50_idx = channel_names.index('z50')
    q50_idx = channel_names.index('q50')
    assert dx.shape[2] == 74, "{}".format(dx.shape)

    dx[:, :, [t50_idx, u50_idx, v50_idx, z50_idx, q50_idx]] = 0
    return dx

def generate_bred_vector(
    x: torch.Tensor,
    model: TimeLoop,
    noise_amplitude: torch.Tensor,
    time: Union[datetime, None] = None,
    integration_steps: int = 5,
    inflate=False,
) -> torch.Tensor:
    # Assume x has shape [ENSEMBLE, TIME, CHANNEL, LAT, LON]

    if isinstance(noise_amplitude, float):
        noise_amplitude = torch.tensor([noise_amplitude]).to(x.device) 
    assert (noise_amplitude.shape[0] == x.shape[2]) or (  # noqa
        torch.numel(noise_amplitude) == 1
    )

    x0 = x[:1]

    # Get control forecast
    for k, (_, data, _) in enumerate(model(time, x0, model_perturb=False)):
        xd = data
        if k == 1:
            break

    # Unsqueeze if time has been collapsed.
    if xd.ndim != x0.ndim:
        xd = xd.unsqueeze(1)

    optimization_target = _load_optimal_targets('sfno_linear_73chq_sc3_layers8_edim384_wstgl2', 6, model.channel_names).to(x.device).squeeze().to(torch.float32) 
    sampler = CorrelatedSphericalField(720, 50. * 1000, 48.0, 1.0, N=74).to(
        x.device
    )
    correlated_random_noise = sampler()
    dx = optimization_target[:, None, None] * correlated_random_noise
    #dx = (correlated_random_noise * 0.05) * x
    channel_idx = set(range(len(model.channel_names))) - set([ model.channel_names.index('z500')])
    dx[:, :, list(channel_idx), : 721, : 1440] = 0
    for _ in range(integration_steps):
        x1 = x + dx
        for k, (_, data, _) in enumerate(model(time, x1, model_perturb=False)):
            x2 = data
            if k == 1:
                break

        # Unsqueeze if time has been collapsed.
        if x2.ndim != x1.ndim:
            x2 = x2.unsqueeze(1)
        dx = x2 - xd

        if inflate:
            dx += noise_amplitude * (dx - dx.mean(dim=0)) * model.scale

    optimization_target = optimization_target[None, None] 
    exaggerate_factor = calculate_global_rms(dx) / optimization_target
    dx = dx / exaggerate_factor[:, :, :, None, None]

    #sampler = CorrelatedSphericalField(720, 500. * 1000, 48.0, 0.002, N=1).to(
    #    x.device
    #)
    #correlated_random_noise = sampler()
    #correlated_random_noise = torch.cat([correlated_random_noise] * 74, dim=1)
    #channels_to_exclude = []
    #for idx, channel in enumerate(model.channel_names):
    #    if channel in ['2d'] or (channel[0] in ['q'] and channel[1:].isdigit() and int(channel[1:]) > 800):
    #        #these channels will be perturbed
    #        pass
    #    else:
    #        #these channels will not be perturbed
    #        channels_to_exclude.append(idx)
    #correlated_random_noise[:, torch.Tensor(np.asarray(channels_to_exclude)).to(torch.int), : 721, : 1440] = 0
    #dx = dx + (x * correlated_random_noise)

    return dx / model.scale 

def calculate_global_geometric_mean(tensor):
    jacobian = torch.sin(
            torch.linspace(0, torch.pi,tensor.shape[-2])
        ).unsqueeze(1).to(tensor.device)
    dtheta = torch.pi / tensor.shape[-2]
    dlambda = 2 * torch.pi / tensor.shape[-1]
    dA = dlambda * dtheta
    quad_weight = dA * jacobian

    weights = quad_weight / quad_weight.mean()
    return (tensor.abs() * weights).log().mean(dim=(-1,-2)).exp()

def calculate_global_rms(tensor):
    jacobian = torch.sin(
            torch.linspace(0, torch.pi,tensor.shape[-2])
        ).unsqueeze(1).to(tensor.device)
    dtheta = torch.pi / tensor.shape[-2]
    dlambda = 2 * torch.pi / tensor.shape[-1]
    dA = dlambda * dtheta
    quad_weight = dA * jacobian

    weights = quad_weight / quad_weight.mean()
    return torch.sqrt(torch.mean(tensor**2 * weights, dim=(-1,-2)))


def calculate_nh_rms(tensor):
    jacobian = torch.sin(
            torch.linspace(0, torch.pi,tensor.shape[-2])
        ).unsqueeze(1).to(tensor.device)
    dtheta = torch.pi / tensor.shape[-2]
    dlambda = 2 * torch.pi / tensor.shape[-1]
    dA = dlambda * dtheta
    quad_weight = dA * jacobian

    weights = quad_weight / quad_weight.mean()
    return torch.sqrt(torch.mean(tensor[:, :, :,:280]**2 * weights[:280], dim=(-1,-2)))


def calculate_sh_rms(tensor):
    jacobian = torch.sin(
            torch.linspace(0, torch.pi,tensor.shape[-2])
        ).unsqueeze(1).to(tensor.device)
    dtheta = torch.pi / tensor.shape[-2]
    dlambda = 2 * torch.pi / tensor.shape[-1]
    dA = dlambda * dtheta
    quad_weight = dA * jacobian

    weights = quad_weight / quad_weight.mean()
    return torch.sqrt(torch.mean(tensor[:, :, :, -280:]**2 * weights[-280:], dim=(-1,-2)))

def calculate_tropics_rms(tensor):
    jacobian = torch.sin(
            torch.linspace(0, torch.pi,tensor.shape[-2])
        ).unsqueeze(1).to(tensor.device)
    dtheta = torch.pi / tensor.shape[-2]
    dlambda = 2 * torch.pi / tensor.shape[-1]
    dA = dlambda * dtheta
    quad_weight = dA * jacobian

    weights = quad_weight / quad_weight.mean()
    return torch.sqrt(torch.mean(tensor[:, :, :, 320:-320]**2 * weights[320:-320], dim=(-1,-2)))

def hemispheric_rms(tensor):
    nh = calculate_nh_rms(tensor).squeeze() 
    sh = calculate_sh_rms(tensor).squeeze()
    t = calculate_tropics_rms(tensor).squeeze()
    hemispheric_rmse = torch.zeros(74, 721, device=sh.device)
    for i in range(74):
        nh_arr = torch.full([280], nh[i])
        sh_arr = torch.full([280], sh[i])
        #tropics = torch.full([161], (nh[i] + sh[i])/2)
        tropics = torch.linspace(nh[i], sh[i], 721 - nh_arr.shape[0]*2) 
        #nh_arr = torch.linspace(nh[i], tropics[0], 240)
        #sh_arr = torch.linspace(tropics[-1], sh[i], 240)
        #tropics = torch.full([161], t[i])
        hemispheric_rmse[i] = torch.cat([nh_arr, tropics, sh_arr])

    return hemispheric_rmse[None, None] 

def graphcast_weights(channel_names, global_stds, loss_type='squared temp-std'):
    channel_weights = torch.ones(len(channel_names), dtype=torch.float32, device=global_stds.device)
    for c, chn in enumerate(channel_names):
        if chn in ["u10m", "v10m", "u100m", "v100m", "tp", "sp", "msl"]:
            channel_weights[c] = 0.1
        elif chn in ["tcwv"]:
            channel_weights[c] = 0.3
        elif chn in ["t2m", "2d", "d2m"]:
            channel_weights[c] = 1.0
        elif chn[0] in ["z", "u", "v", "t", "r", "q"]:
            pressure_level = float(chn[1:])
            channel_weights[c] = 0.001 * pressure_level
        else:
            channel_weights[c] = 0.01

    # renormalize the weights to one
    channel_weights = channel_weights.reshape(1, -1, 1, 1)
    #channel_weights = channel_weights / torch.sum(channel_weights)

    if "squared" in loss_type:
        squared = True
    else:
        squared = False

    target =  _load_optimal_targets("", 24, channel_names).squeeze().to(channel_weights.device) 
    if "temp-std" in loss_type:
        eps = 1e-6
        global_stds = global_stds.reshape(1, -1, 1, 1)
        dt = 1
        time_diff_stds = np.sqrt(dt) * torch.from_numpy(np.load("/pscratch/sd/p/pharring/74var-6hourly/staging/stats_fcndev/time_diff_stds.npy")).reshape(1, -1, 1, 1).to(global_stds.device)
        time_var_weights = global_stds / (time_diff_stds + eps)
        # time_var_weights = 1 / (time_diff_stds+eps)
        if squared:
            time_var_weights = time_var_weights**2
            #target = target**2
        channel_weights = channel_weights * time_var_weights
    if "deterministic rmse" in loss_type:
        channel_weights = channel_weights.squeeze() * target
    return channel_weights

def _load_optimal_targets(config_model_name, lead_time, channel_names):
    from earth2mip import forecast_metrics_io
    sfno = xr.open_dataset("/pscratch/sd/a/amahesh/hens/optimal_perturbation_targets/means/d2m_sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed16.nc")
    targets = []
    #return torch.from_numpy(sfno['value'].sel(channel=channel_names).sel(lead_time=lead_time).values[np.newaxis])
    for name in channel_names:
        if name[0] in ['u', 'v', 't', 'q', 'd', '2']:
            lead_time = lead_time
        else:
            lead_time = lead_time
        targets.append(sfno['value'].sel(channel=name, lead_time=lead_time).values)
        #if name == 'z50':
        #    targets.append(sfno['value'].sel(channel='z100', lead_time=lead_time).values)
        #else:
        #    targets.append(sfno['value'].sel(channel=name, lead_time=lead_time).values)
    return torch.Tensor(np.asarray(targets))[None]



def generate_model_noise_correlated(
    x,
    time,
    scale,
    reddening,
    noise_injection_amplitude,
):
    noise = noise_injection_amplitude * brown_noise(x.shape, reddening).to(x.device)
    channels = list(set(range(73)) - set([41]))
    noise[:, :, channels, : 721, : 1440] = 0
    return noise * scale / 21600

def generate_multiplicative_noise(x, time, length_scale, channel_names, sht, ivsht, weighing_method="zero", eps=50, std=0.18, center=1):
    """                                                                 
    Inputs                                                              
                                                                        
        length_scale (int): length scale in km of spherical random noise
        weighing_method (str) : how to weigh the wavelengths outside of the weighing method
                                "zero" to zero them out                 
                                "gaussian" for gaussian weighting with mean 0 and sd 0.01
                                "inverse" for weighing them by their inverse distance from length_scale
        eps (int).        : the number of km plus or minus length scale to provide full weight to
                                                                        
    """                                                                 
    shape = (x.shape[-2], x.shape[-1])
    device = x.device
    rand_noise = torch.randn(*shape, device=device)                     
    nlat, nlon = rand_noise.shape                                       
                                                                        
    coeffs = sht(rand_noise)                                            
    r_earth = 6.371e6                                                   
                                                                        
    l = torch.arange(nlat, device=device)
    # convert spherical harmonic degree to wavelength [km]              
    lam = 2*np.pi*r_earth/(l + 1) / 1e3                                 
                                                                        
    #How to weigh the wavelengths not in the range of length scale      
    if weighing_method == "gaussian":                                   
        weights = np.random.normal(0, 0.01, size=coeffs.shape)          
    elif weighing_method == "zero":                                     
        weights = torch.zeros(coeffs.shape, device=device) 
    else:                                                               
        assert NotImplemented                                           

    #epsilon
    weights[torch.logical_and(lam >= length_scale - eps, lam < length_scale + eps), : ] = 1
    coeffs_weighted = coeffs * weights

    correlated_noise = ivsht(coeffs_weighted)
    correlated_noise = (correlated_noise - correlated_noise.mean())/(correlated_noise.std()) * std
    #if grid != schema.Grid.grid_721x1440:
    #    correlated_noise = correlated_noise[0:-1, :]  # 720 x 1440
    
    num_channels = len(channel_names)
    channels_to_exclude = []
    stacked_noise = torch.stack([correlated_noise]*num_channels, dim=0)[None].to(torch.float32)
    for idx, channel in enumerate(channel_names):
        if channel[0] in ['u', 'v', 't', 'q'] and channel[1:].isdigit():
            pass
        else:
            channels_to_exclude.append(idx)
    
    print('model tendency perturbed')
    stacked_noise[:, channels_to_exclude, : 721, : 1440] = 0
    stacked_noise = torch.where(stacked_noise > 1, 1, stacked_noise)
    stacked_noise = torch.where(stacked_noise < -1, -1, stacked_noise)
    return stacked_noise + center

class CorrelatedSphericalField(torch.nn.Module):
    def __init__(
        self,
        nlat,
        length_scale,
        time_scale,
        sigma,
        N=1,
        channel_names=None,
        grid="equiangular",
        dtype=torch.float32,
    ):
        super().__init__()
        """
        This class can be used to create noise on the sphere
        with a given length scale (in km) and time scale (in hours).

        It mimics the implementation of the SPPT: Stochastic Perturbed
        Parameterized Tendency in this paper:

        https://www.ecmwf.int/sites/default/files/elibrary/2009/11577-stochastic-parametrization-and-model-uncertainty.pdf

        Parameters
        ----------
        length_scale : int
            Length scale in km

        time_scale : int
            Time scale for the AR(1) process, that governs
            the evolution of the coefficients

        sigma: desired standard deviation of the field in
                grid point space

        nlat : int
            Number of latitudinal modes;
            longitudinal modes are 2*nlat.
        grid : string, default is "equiangular"
            Grid type. Currently supports "equiangular" and
            "legendre-gauss".
        dtype : torch.dtype, default is torch.float32
            Numerical type for the calculations.
        """
        self.sigma = sigma
        self.channel_names = channel_names
        dt = 6.0
        self.phi = np.exp(-dt/time_scale)

        # Number of latitudinal modes.
        self.nlat = nlat

        # Inverse SHT
        self.isht = th.InverseRealSHT(
            self.nlat, 2 * self.nlat, grid=grid, norm="backward"
        ).to(dtype=dtype)

        r_earth = 6.371e6
        #kT is defined on slide 7
        self.kT = (length_scale/r_earth)**2 / 2
        F0 = self.calculateF0(self.sigma, self.phi, self.nlat, self.kT)

        prods = (
            torch.tensor([j * (j + 1) for j in range(0,self.nlat)])
            .view(self.nlat, 1)
            .repeat(1, self.nlat + 1)
        )


        sigma_n = torch.tril(torch.exp(-self.kT * prods / 2) * F0)
        self.register_buffer("sigma_n", sigma_n)

        # Save mean and var of the standard Gaussian.
        # Need these to re-initialize distribution on a new device.
        mean = torch.tensor([0.0]).to(dtype=dtype)
        var = torch.tensor([1.0]).to(dtype=dtype)
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)
        self.N = N

        # Standard normal noise sampler.
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)
        xi = self.gaussian_noise.sample(
                torch.Size((self.N, self.nlat, self.nlat + 1, 2))
            ).squeeze()
        xi = torch.view_as_complex(xi)

        #Set specrtral cofficients to this value at initial time
        #for stability in teh AR(1) process.  See link in description
        coeff = ((1-self.phi**2)**(-0.5)) * self.sigma_n * xi
        coeff = coeff.unsqueeze(0)
        self.register_buffer("coeff", coeff)

    def calculateF0(self, sigma, phi, nlat, kT):
        """
            This function scales the coefficients such that their
            grid-point standard deviation is sigma.
            sigma is the desired variance
            phi is a np.exp(-dt/time_scale)
        """
        numerator = sigma**2 * (1-(phi**2))
        wavenumbers = torch.arange(1,nlat)
        denominator = (2 * wavenumbers + 1) * torch.exp(-kT * wavenumbers * (wavenumbers+1))
        denominator = 2 * denominator.sum()
        return (numerator / denominator)**0.5

    def forward(self, x=None, time=None):
        """
        Generate and return a field with a correlated length scale.

        Update the coefficients using an AR(1) process.
        """
        u = self.isht(self.coeff) * 4 * np.pi
        u = u.reshape(1, self.N, self.nlat, self.nlat * 2)
        
        noise = torch.zeros((1, self.N, self.nlat+1, self.nlat * 2), device=u.device)
        noise[ :, :, :-1, :] = u
        noise[ :, :, -1:, :] = noise[ :, :, -2:-1, :]

        # Sample Gaussian noise.
        xi = self.gaussian_noise.sample(
            torch.Size((self.N, self.nlat, self.nlat + 1, 2))
        ).squeeze()
        xi = torch.view_as_complex(xi)

        self.coeff = (self.phi * self.coeff) + (self.sigma_n * xi)
        if x is not None:
            num_channels = len(self.channel_names)
            channels_to_exclude = []
            noise = torch.cat([noise]*num_channels,dim=1)
            for idx, channel in enumerate(self.channel_names):
                #if channel[0] in ['u', 'v', 't', 'q'] and channel[1:].isdigit():
                if channel in ['2d', 'd2m'] or (channel[0] in ['q'] and channel[1:].isdigit() and int(channel[1:]) > 800):
                    #these channels will be perturbed
                    pass
                else:
                    #these channels will not be perturbed
                    channels_to_exclude.append(idx)
            
            print('model tendency perturbed: all channels except: {}'.format(np.asarray(self.channel_names)[channels_to_exclude]))
            #noise[:, torch.Tensor(np.asarray(channels_to_exclude)).to(torch.int), : 721, : 1440] = 0
            #noise= torch.where(noise > 1, 1, noise)
            #noise = torch.where(noise < -1, -1, noise)
            noise = noise + 1
            #stacked_noise = stacked_noise + 1
            #noise = stacked_noise
        return noise

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

