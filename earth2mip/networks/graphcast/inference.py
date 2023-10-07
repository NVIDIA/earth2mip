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
from typing import Literal
import dataclasses
import datetime
import functools
import os

import einops
import haiku as hk
import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
import xarray
from graphcast import checkpoint, data_utils, graphcast

from earth2mip import schema
from earth2mip.initial_conditions import cds
from earth2mip.networks.graphcast import channels
from earth2mip.time_loop import TimeLoop

__all__ = ["load_time_loop"]


def torch_to_jax(x):
    return jax.dlpack.from_dlpack(torch.to_dlpack(x))


class NoXarrayGraphcast(graphcast.GraphCast):
    def __call__(self, grid_node_features, lat, lon):
        # Transfer data for the grid to the mesh,
        # [num_mesh_nodes, batch, latent_size], [num_grid_nodes, batch, latent_size]
        if not self._initialized:
            self._init_mesh_properties()
            self._init_grid_properties(grid_lat=lat, grid_lon=lon)
            self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
            self._mesh_graph_structure = self._init_mesh_graph()
            self._mesh2grid_graph_structure = self._init_mesh2grid_graph()

            self._initialized = True

        (latent_mesh_nodes, latent_grid_nodes) = self._run_grid2mesh_gnn(
            grid_node_features
        )

        # Run message passing in the multimesh.
        # [num_mesh_nodes, batch, latent_size]
        updated_latent_mesh_nodes = self._run_mesh_gnn(latent_mesh_nodes)

        # Transfer data frome the mesh to the grid.
        # [num_grid_nodes, batch, output_size]
        output_grid_nodes = self._run_mesh2grid_gnn(
            updated_latent_mesh_nodes, latent_grid_nodes
        )

        return output_grid_nodes


def load_run_forward_from_checkpoint(checkpoint, grid):
    state = {}
    params = checkpoint.params
    model_config = checkpoint.model_config
    task_config = checkpoint.task_config
    print("Model description:\n", checkpoint.description, "\n")
    print("Model license:\n", checkpoint.license, "\n")

    @hk.transform_with_state
    def run_forward(model_config, task_config, x):
        lat = grid.lat[::-1]
        lon = grid.lon

        x = x.astype(jnp.float16)
        predictor = NoXarrayGraphcast(model_config, task_config)
        return predictor(x, lat, lon)

    # Jax doesn't seem to like passing configs as args through the jit. Passing it
    # in via partial (instead of capture by closure) forces jax to invalidate the
    # jit cache if you change configs.
    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    # Always pass params and state, so the usage below are simpler
    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    # Our models aren't stateful, so the state is always empty, so just return the
    # predictions. This is requiredy by our rollout code, and generally simpler.
    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    return drop_state(with_params(jax.jit(with_configs(run_forward.apply))))


class GraphcastTimeLoop(TimeLoop):
    """
    # packing notes
    # to graph inputs
    # 1. b h c y x -> (y x) b (h c)
    # 2. normalize
    # 3. x = cat([in, f])
    # 4. y = denorm(f(x)) + x

    """

    grid = schema.Grid.grid_721x1440
    n_history_levels: int = 2
    history_time_step: datetime.timedelta = datetime.timedelta(hours=6)
    time_step: datetime.timedelta = datetime.timedelta(hours=6)
    dtype: torch.dtype = torch.float32

    def __init__(
        self,
        forward,
        static_variables,
        mean,
        scale,
        diff_scale,
        task_config,
        # TODO move into grid
        lat: np.ndarray,
        lon: np.ndarray,
    ):
        in_codes, t_codes = channels.get_codes_from_task_config(task_config)
        self.lon = lon
        self.lat = lat
        self.task_config = task_config
        self.forward = forward
        self._static_variables = static_variables
        self.mean = mean
        self.scale = scale
        self.diff_scale = diff_scale
        self.in_codes = in_codes
        self.target_codes = t_codes

        self.prog_levels = [
            [in_codes.index(k) for k in channels.get_state_codes(task_config, t)]
            for t in range(2)
        ]

        self.in_channel_names = [
            str(c) for _, c in channels.get_state_codes(task_config, 0)
        ]

        # setup output names
        state_codes = channels.get_state_codes(self.task_config, 0)
        state_names = [str(c) for _, c in state_codes]
        self._diagnostic_names = [
            str(c) for _, c in self.target_codes if str(c) not in state_names
        ]
        self.out_channel_names = state_names + self._diagnostic_names

    def set_static(self, array, field, arr):
        k = self.in_codes.index(field)
        array[:, 0, k] = einops.rearrange(arr, "y x ->  (y x)")

    def set_static_variables(self, array):
        for field, arr in self._static_variables.items():
            arr = torch.from_numpy(arr)
            self.set_static(array, field, arr)

    def set_forcing(self, array, v, t, data):
        # (y x) b c
        i = self.in_codes.index((t, v))
        return array.at[:, :, i].set(data)

    def set_forcings(self, x, time: datetime.datetime, t: int):
        seconds = time.timestamp()
        lat, lon = np.meshgrid(self.lat, self.lon, indexing="ij")

        lat = lat.reshape([-1, 1])
        lon = lon.reshape([-1, 1])

        day_progress = data_utils.get_day_progress(seconds, lon)
        year_progress = data_utils.get_year_progress(seconds)
        x = self.set_forcing(x, "day_progress_sin", t, np.sin(day_progress))
        x = self.set_forcing(x, "day_progress_cos", t, np.cos(day_progress))
        x = self.set_forcing(x, "year_progress_sin", t, np.sin(year_progress))
        x = self.set_forcing(x, "year_progress_cos", t, np.cos(year_progress))

        tisr = channels.toa_incident_solar_radiation(time, lat, lon)
        return self.set_forcing(x, "toa_incident_solar_radiation", t, tisr)

    def set_prognostic(self, array, t: int, data):
        index = self.prog_levels[t]
        return array.at[:, :, index].set(data)

    def get_prognostic(self, array, t: int):
        index = self.prog_levels[t]
        return array[:, :, index]

    def split_target(self, target):
        state_codes = channels.get_state_codes(self.task_config, 0)
        index = np.array([c in state_codes for c in self.target_codes])

        state_increment = target[:, :, index]
        diagnostics = target[:, :, ~index]
        return state_increment, diagnostics

    def _to_latlon(self, array):
        array = einops.rearrange(array, "(y x) b c -> b c y x", y=len(self.lat))
        p = jax.dlpack.to_dlpack(array)
        pt = torch.from_dlpack(p)
        return torch.flip(pt, [-2])

    def _input_codes(self):
        return list(get_codes(self.task_config))

    def step(self, rng, time, s):
        s = self.set_forcings(s, time - 1 * self.history_time_step, 0)
        s = self.set_forcings(s, time, 1)
        s = self.set_forcings(s, time + self.history_time_step, 2)

        x = (s - self.mean) / self.scale
        # TODO rename to target scale
        d = self.forward(rng=rng, x=x) * self.diff_scale
        diff, diagnostics = self.split_target(d)
        x_next = self.get_prognostic(s, 1) + diff

        # update array
        s = self.set_prognostic(s, 0, self.get_prognostic(s, 1))
        s = self.set_prognostic(s, 1, x_next)

        # add state to output diagnostics
        diagnostics = jnp.concatenate([x_next, diagnostics], axis=-1)
        return s, diagnostics

    def __call__(self, time, x, restart=None):
        assert not restart, "not implemented"
        ngrid = len(self.lon) * len(self.lat)
        array = torch.empty([ngrid, 1, len(self.in_codes)], device=x.device)

        # set input data
        x_codes = [cds.parse_channel(name) for name in self.in_channel_names]
        for t in range(2):
            index_in_input = [self.in_codes.index((t, c)) for c in x_codes]
            array[:, :, index_in_input] = einops.rearrange(
                torch.flip(x[:, t], [-2]), "b c y x -> (y x) b c"
            )

        self.set_static_variables(array)

        rng = jax.random.PRNGKey(0)
        s = torch_to_jax(array)

        # TODO fill in state for the first time step
        diagnostics = jnp.full([ngrid, x.shape[0], len(self.out_channel_names)], np.nan)
        while True:
            yield time, self._to_latlon(diagnostics), None
            s, diagnostics = self.step(rng, time, s)
            time = time + self.time_step


def get_codes(task_config):
    codes = []
    levels = list(task_config.pressure_levels)
    lookup = cds.keys_to_vals(channels.CODE_TO_GRAPHCAST_NAME)
    for v in task_config.target_variables:
        id = lookup[v]
        if channels.is_3d(v):
            for lev in levels:
                yield cds.PressureLevelCode(id, level=lev)
        else:
            yield cds.SingleLevelCode(id)
    return codes


@dataclasses.dataclass
class GraphcastDescription:
    checkpoint: str
    resolution: float
    nlevels: int


def get_static_data(package, resolution):
    dataset_location = {
        0.25: "dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc",
        1.0: "dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc",
    }[resolution]

    static_data_path = package.get(dataset_location)

    with open(static_data_path, "rb") as f:
        example_batch = xarray.load_dataset(f).compute()
    return {
        key: example_batch[key].values
        for key in ["land_sea_mask", "geopotential_at_surface"]
    }


def load_time_loop(
    package,
    pretrained=True,
    device="cuda:0",
    version: Literal["paper", "operational", "small"] = "paper",
):
    def join(*args):
        return package.get(os.path.join(*args))

    models = {
        "paper": GraphcastDescription(
            "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz",
            0.25,
            37,
        ),
        "operational": GraphcastDescription(
            "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz",
            0.25,
            13,
        ),
        # TODO for small need to add a new grid
        "small": GraphcastDescription(
            "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
            1.0,
            13,
        ),
    }
    model = models[version]

    checkpoint_path = join("params", model.checkpoint)
    # load checkpoint:
    with open(checkpoint_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
        task_config = ckpt.task_config
        run_forward = load_run_forward_from_checkpoint(
            ckpt, grid=GraphcastTimeLoop.grid
        )

    static_variables = get_static_data(package, model.resolution)

    # load stats
    with open(join("stats/diffs_stddev_by_level.nc"), "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open(join("stats/mean_by_level.nc"), "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open(join("stats/stddev_by_level.nc"), "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()

    # select needed channels from stats
    in_codes, t_codes = channels.get_codes_from_task_config(task_config)
    mean = np.array(
        [channels.get_data_for_code_scalar(code, mean_by_level) for code in in_codes]
    )
    scale = np.array(
        [channels.get_data_for_code_scalar(code, stddev_by_level) for code in in_codes]
    )
    diff_scale = np.array(
        [
            channels.get_data_for_code_scalar(code, diffs_stddev_by_level)
            for code in t_codes
        ]
    )
    return GraphcastTimeLoop(
        run_forward,
        static_variables,
        mean,
        scale,
        diff_scale,
        task_config,
        lat=GraphcastTimeLoop.grid.lat,
        lon=GraphcastTimeLoop.grid.lon,
    )
