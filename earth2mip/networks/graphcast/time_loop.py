from typing import Iterator, Tuple, Any, Optional
from earth2mip.networks.graphcast.implementation import GraphcastStepper

import torch
import jax.dlpack
import pandas as pd
from earth2mip import time_loop
import earth2mip.grid
import jax
import numpy as np
import datetime


class GraphcastTimeLoop(time_loop.TimeLoop):
    def __init__(self, stepper: GraphcastStepper, device=None):
        self.stepper = stepper
        self.grid = earth2mip.grid.LatLonGrid(
            self.stepper.lat.tolist(), self.stepper.lon.tolist()
        )
        self.history_time_step = pd.Timedelta("6h")
        self.time_step = pd.Timedelta("6h")
        self.n_history_levels = (
            pd.Timedelta(stepper.task_config.input_duration) // self.history_time_step
        )
        self.device = device or torch.cuda.current_device()
        self.dtype = torch.float32

    @property
    def in_channel_names(self):
        return self.stepper.get_in_channel_names()

    @property
    def out_channel_names(self):
        return self.in_channel_names

    def __call__(
        self, time: datetime.datetime, x: torch.Tensor, restart: Optional[Any] = None
    ) -> Iterator[Tuple[datetime.datetime, torch.Tensor, Any]]:
        """
        Args:
            x: an initial condition. has shape (B, n_history_levels,
                len(in_channel_names), Y, X).  (Y, X) should be consistent with
                ``grid``. The history dimension is in increasing order, so the
                current state corresponds to x[:, -1].  Specifically, ``x[:,
                -i]`` is the data correspond to ``time - (i-1) *
                self.history_time_step``.
            time: the datetime to start with, by default assumed to be in UTC.
            restart: if provided this restart information (typically some torch
                Tensor) can be used to restart the time loop

        Yields:
            (time, output, restart) tuples. ``output`` is a tensor with
                shape (B, len(out_channel_names), Y, X) which will be used for
                diagnostics. Restart data should encode the state of the time
                loop.
        """
        assert not restart
        rng = jax.random.PRNGKey(0)
        time = pd.Timestamp(time)
        dt = pd.Timedelta("6h")
        x_jax = torch_to_jax(x)
        yield time.to_pydatetime(), x[:, -1], None
        next = self.stepper.get_inputs(x_jax, time, dt)
        while True:
            # TODO also return outputs like tp06
            time, next, rng = self.stepper.step(time, next, rng)
            assert not np.any(np.isnan(next)).to_array().any()

            array = self.stepper.pack(next)
            output = jax_to_torch(array)
            assert output.ndim == 5
            yield time.to_pydatetime(), output[:, -1], next


def jax_to_torch(x):
    return torch.from_dlpack(jax.dlpack.to_dlpack(x))


def torch_to_jax(x):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))
