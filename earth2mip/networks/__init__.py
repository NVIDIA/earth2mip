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

import pathlib
import urllib
import warnings
import itertools
from typing import Optional, Tuple, Any, Iterator
import sys
import datetime
import os

import torch
import einops
import numpy as np
import contextlib
import modulus


from modulus.utils.sfno.zenith_angle import cos_zenith_angle
from modulus.distributed.manager import DistributedManager
from earth2mip.loaders import LoaderProtocol
from earth2mip import registry, ModelRegistry, model_registry
from earth2mip import filesystem, loaders, time_loop, schema

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint, entry_points
else:
    from importlib.metadata import EntryPoint, entry_points


__all__ = ["get_model"]


def depends_on_time(f):
    """
    A function to detect if the function `f` takes an argument `time`.

    Args:
        f: a function.

    Returns:
        bool: True if the function takes a second argument `time`, False otherwise.
    """
    # check if model is a torchscript model
    if isinstance(f, torch.jit.ScriptModule):
        return False
    else:
        import inspect

        signature = inspect.signature(f)
        parameters = signature.parameters
        return "time" in parameters


class Wrapper(torch.nn.Module):
    """Makes sure the parameter names are the same as the checkpoint"""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        """x: (batch, history, channel, x, y)"""
        return self.module(*args, **kwargs)


class CosZenWrapper(torch.nn.Module):
    def __init__(self, model, lon, lat):
        super().__init__()
        self.model = model
        self.lon = lon
        self.lat = lat

    def forward(self, x, time):
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        cosz = cos_zenith_angle(time, lon_grid, lat_grid)
        cosz = cosz.astype(np.float32)
        z = torch.from_numpy(cosz).to(device=x.device)
        # assume no history
        x = torch.cat([x, z[None, None]], dim=1)
        return self.model(x)


class _SimpleModelAdapter(torch.nn.Module):
    """Takes model of (b, c, y, x) to (b, h, y, x) where h == 1"""

    def __init__(self, model, time_dependent, has_history):
        super().__init__()
        self.model = model
        self.time_dependent = time_dependent
        self.has_history = has_history

    def forward(self, x, time):
        if not self.has_history:
            x = x[:, 0]

        if self.time_dependent:
            y = self.model.forward(x, time)
        else:
            y = self.model.forward(x)

        if not self.has_history:
            y = y[:, None]

        return y


class Inference(torch.nn.Module, time_loop.TimeLoop):
    def __init__(
        self,
        model,
        center: np.array,
        scale: np.array,
        grid: schema.Grid,
        channels=None,
        channel_set: Optional[schema.ChannelSet] = None,
        n_history: int = 0,
        time_step=datetime.timedelta(hours=6),
        channel_names=None,
    ):
        """
        Args:
            model: a model, with signature model(x, time) or model(x). With n_history == 0, x is a
                torch tensor with shape (batch, nchannel, lat, lon). With
                n_history > 0 x has the shape (batch, nchannel, lat, lon).
                `time` is a datetime object, which is passed if model.forward has time as an argument.
            center: a 1d numpy array with shape (n_channels in data) containing
                the means. The shape is NOT `len(channels)`.
            scale: a 1d numpy array with shape (n_channels in data) containing
                the stds. The shape is NOT `len(channels)`.
            grid: metadata about the grid, which should be used to pass the
                correct data to this object.
            channels: a list of integers taken from [0, n_channels in data -
                1]. This is used to subset the input data.
            channel_set: optional metadata about the channel-set, that can be
                used to figure out what the channel names are.
            channel_names: if provided overrides the ``channel_set`` and
                ``channels``. If this is provided then mean/scale are assumed to
                match this.
            n_history: whether `model` was trained with history.
            time_step: the time-step `model` was trained with.

        """  # noqa
        super().__init__()
        self.channel_set = channel_set
        self.time_dependent = depends_on_time(model.forward)

        # TODO probably delete this line
        # if not isinstance(model, modulus.Module):
        #     model = Wrapper(model)

        # TODO extract this to another place
        model = _SimpleModelAdapter(
            model, time_dependent=self.time_dependent, has_history=n_history > 0
        )

        self.model = model
        self.channel_set = channel_set
        self.channel_names = channel_names
        self.grid = grid
        self.time_step = time_step
        self.n_history = n_history

        center = torch.from_numpy(np.squeeze(center)).float()
        scale = torch.from_numpy(np.squeeze(scale)).float()
        self.register_buffer("scale_org", scale)
        self.register_buffer("center_org", center)

        # infer channel names
        if channel_names is not None:
            self.in_channel_names = self.out_channel_names = channel_names
            self.channels = list(range(len(channel_names)))
            self.register_buffer("scale", scale[:, None, None])
            self.register_buffer("center", center[:, None, None])
        elif channel_set is not None:
            data_channel_names = channel_set.list_channels()
            self.channels = channels
            self.in_channel_names = self.out_channel_names = [
                data_channel_names[i] for i in channels
            ]
            self.register_buffer("scale", scale[self.channels, None, None])
            self.register_buffer("center", center[self.channels, None, None])
        else:
            raise ValueError(
                "Cannot infer channel names. Please provide channel_names or both channels and channel_set."  # noqa
            )

    @property
    def n_history_levels(self) -> int:
        """The expected size of the second dimension"""
        return self.n_history + 1

    def device(self) -> torch.device:
        return self.scale.device

    def normalize(self, x):
        warnings.warn(
            DeprecationWarning(
                ".normalize does not follow the TimeLoop API. It will be removed soon."
            )
        )
        return (x - self.center_org[None, :, None, None]) / self.scale_org[
            None, :, None, None
        ]

    def run_steps(self, x, n, normalize=True, time=None):
        warnings.warn(
            DeprecationWarning(
                ".run_steps does not follow the TimeLoop API. It will be removed soon."
            )
        )
        for _, data, _ in self.run_steps_with_restart(x, n, normalize, time):
            yield data

    def __call__(
        self,
        time: datetime.datetime,
        x: torch.Tensor,
        restart: Optional[Any] = None,
        normalize=True,
    ) -> Iterator[Tuple[datetime.datetime, torch.Tensor, Any]]:
        """
        Args:
            x: an initial condition. has shape (B, n_history_levels,
                len(in_channel_names), Y, X).  (Y, X) should be consistent with
                ``grid``.
            time: the datetime to start with
            restart: if provided this restart information (typically some torch
                Tensor) can be used to restart the time loop

        Yields:
            (time, output, restart) tuples. ``output`` is a tensor with
                shape (B, len(out_channel_names), Y, X) which will be used for
                diagnostics. Restart data should encode the state of the time
                loop.
        """
        if restart:
            yield from self._iterate(**restart)
        else:
            yield from self._iterate(x=x, time=time, n=None, normalize=normalize)

    def run_steps_with_restart(self, x, n, normalize=True, time=None):
        warnings.warn(
            DeprecationWarning(
                ".run_steps_with_restart does not follow the TimeLoop API. It will be removed soon."  # noqa
            )
        )
        x = x[:, :, self.channels]
        yield from self._iterate(x, n, normalize=normalize, time=time)

    def _iterate(self, x, n, normalize=True, time=None):
        """Yield (time, unnormalized data, restart) tuples

        restart = (time, unnormalized data)
        """
        if self.time_dependent and not time:
            raise ValueError("Time dependent models require ``time``.")

        time = time or datetime.datetime(1900, 1, 1)
        with torch.no_grad():
            # drop all but the last time point
            # remove channels

            _, n_time_levels, n_channels, _, _ = x.shape
            assert n_time_levels == self.n_history + 1

            if normalize:
                x = (x - self.center) / self.scale

            # yield initial time for convenience
            restart = dict(x=x, normalize=False, time=time)
            yield time, self.scale * x[:, -1] + self.center, restart

            for i in range(n) if n else itertools.count():
                x = self.model(x, time)
                time = time + self.time_step

                # create args and kwargs for future use
                restart = dict(x=x, normalize=False, time=time)
                out = self.scale * x[:, -1] + self.center
                yield time, out, restart


def _default_inference(package, metadata, device):
    if metadata.architecture == "pickle":
        loader = loaders.pickle
    elif metadata.architecture_entrypoint:
        ep = EntryPoint(name=None, group=None, value=metadata.architecture_entrypoint)
        loader: LoaderProtocol = ep.load()
    else:
        raise NotImplementedError()

    model = loader(package, pretrained=True)

    center_path = package.get("global_means.npy")
    scale_path = package.get("global_stds.npy")

    inference = Inference(
        model=model,
        channels=metadata.in_channels,
        center=np.load(center_path),
        scale=np.load(scale_path),
        grid=metadata.grid,
        channel_set=metadata.channel_set,
        n_history=metadata.n_history,
        time_step=metadata.time_step,
    )
    inference.to(device)
    return inference


def _load_package(package, metadata, device) -> time_loop.TimeLoop:
    # Attempt to see if Earth2 MIP has entry point registered already
    if metadata is None:
        group = "earth2mip.networks"
        entrypoints = entry_points(group=group)
        for entry_point in entrypoints:
            name = package.root.split(package.seperator)[-1]
            if entry_point.name == name:
                inference_loader = entry_point.load()
                return inference_loader(package, device=device)

    # Read meta data from file if not present
    if metadata is None:
        local_path = package.get("metadata.json")
        with open(local_path) as f:
            metadata = schema.Model.parse_raw(f.read())

    if metadata.entrypoint:
        ep = EntryPoint(name=None, group=None, value=metadata.entrypoint.name)
        inference_loader = ep.load()
        return inference_loader(package, device=device, **metadata.entrypoint.kwargs)
    else:
        warnings.warn("No loading entry point found, using default inferencer")
        return _default_inference(package, metadata, device=device)


def get_model(
    model: str,
    registry: ModelRegistry = registry,
    device="cpu",
    metadata: Optional[schema.Model] = None,
) -> time_loop.TimeLoop:
    """
    Function to construct an inference model and load the appropriate
    checkpoints from the model registry

    Parameters
    ----------
    model : The model name to open in the ``registry``. If a url is passed (e.g.
        s3://bucket/model), then this location will be opened directly.
        Supported urls protocols include s3:// for PBSS access, and file:// for
        local files.
    registry: A model registry object. Defaults to the global model registry
    metadata: If provided, this model metadata will be used to load the model.
        By default this will be loaded from the file ``metadata.json`` in the
        model package.
    device: the device to load on, by default the 'cpu'


    Returns
    -------
    Inference model


    """
    url = urllib.parse.urlparse(model)
    if url.scheme == "" or url.scheme == "e2mip":
        package = registry.get_model(model)
    else:
        package = model_registry.Package(root=model, seperator="/")

    return _load_package(package, metadata, device)


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def persistence(package, pretrained=True):
    model = Identity()
    center = np.zeros((3))
    scale = np.zeros((3))
    grid = schema.Grid.grid_721x1440
    return Inference(
        model,
        channel_names=["a", "b", "c"],
        center=center,
        scale=scale,
        grid=grid,
        n_history=0,
    )
