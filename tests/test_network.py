from earth2mip import networks, schema
import torch
import torch.nn
import numpy as np


class Identity(torch.nn.Module):
    def forward(self, x):
        return x + 0.01


def test_inference_run_with_restart():
    model = Identity()
    channels = [0, 1]
    center = [0, 0]
    scale = [1, 1]

    # batch, time_levels, channels, y, x
    x = torch.zeros([1, 1, 2, 5, 6])
    model = networks.Inference(
        model,
        center=center,
        channels=channels,
        scale=scale,
        grid=schema.Grid.grid_720x1440,
        channel_set=schema.ChannelSet.var34,
    )

    step1 = []
    for _, state, restart in model.run_steps_with_restart(x, 3):
        step1.append(restart)
    assert len(step1) == 4

    # start run from 50% done
    for _, final_state, _ in model.run_steps_with_restart(n=2, **step1[1]):
        pass

    np.testing.assert_array_equal(final_state.numpy(), state.numpy())
