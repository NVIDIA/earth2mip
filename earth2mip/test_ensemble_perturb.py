import torch
import pytest
import datetime

from earth2mip.ensemble_utils import generate_noise_correlated, generate_bred_vector
from earth2mip import networks
from earth2mip.schema import Grid, ChannelSet


@pytest.mark.slow
def test_generate_noise_correlated():
    torch.manual_seed(0)
    shape = (2, 34, 32, 64)
    noise = generate_noise_correlated(
        shape=shape, reddening=2.0, noise_amplitude=0.1, device="cpu"
    )
    assert tuple(noise.shape) == tuple(shape)
    assert torch.mean(noise) < torch.tensor(1e-09).to()


class Dummy(torch.nn.Module):
    def forward(self, x, time):
        return 2.5 * torch.abs(x) * (1 - torch.abs(x))


def test_bred_vector():
    device = "cpu"
    model = Dummy().to(device)
    initial_time = datetime.datetime(2018, 1, 1)
    channels = [0, 1]
    center = [0, 0]
    scale = [1, 1]

    # batch, time_levels, channels, y, x
    x = torch.rand([4, 1, 2, 5, 6], device=device)
    model = networks.Inference(
        model,
        center=center,
        channels=channels,
        scale=scale,
        grid=Grid.grid_720x1440,
        channel_set=ChannelSet.var34,
    ).to(device)

    noise_amplitude = 0.01
    noise = generate_bred_vector(
        x,
        model,
        noise_amplitude=noise_amplitude,
        time=initial_time,
        integration_steps=20,
        inflate=False,
    )
    assert noise.device == x.device
    assert noise.shape == x.shape
    assert not torch.any(torch.isnan(noise))

    noise = generate_bred_vector(
        x,
        model,
        noise_amplitude=noise_amplitude,
        time=initial_time,
        integration_steps=20,
        inflate=True,
    )
    assert noise.device == x.device
    assert noise.shape == x.shape
    assert not torch.any(torch.isnan(noise))
