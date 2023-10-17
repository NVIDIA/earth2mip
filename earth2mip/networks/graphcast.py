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
import dataclasses
import datetime
import functools
import os
from typing import List, Literal

import einops
import haiku as hk
import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
import torch
import xarray
from graphcast import checkpoint, data_utils, graphcast
from graphcast.graphcast import TaskConfig
import pytz
from modulus.utils.zenith_angle import toa_incident_solar_radiation_accumulated

from earth2mip import schema
from earth2mip.initial_conditions import cds
from earth2mip.time_loop import TimeLoop
import earth2mip.time

import logging

__all__ = ["load_time_loop", "load_time_loop_operational", "load_time_loop_small"]

# see ecwmf parameter table https://codes.ecmwf.int/grib/param-db/?&filter=grib1&table=128
CODE_TO_GRAPHCAST_NAME = {
    167: "2m_temperature",
    151: "mean_sea_level_pressure",
    166: "10m_v_component_of_wind",
    165: "10m_u_component_of_wind",
    260267: "total_precipitation_6hr",
    212: "toa_incident_solar_radiation",
    130: "temperature",
    129: "geopotential",
    131: "u_component_of_wind",
    132: "v_component_of_wind",
    135: "vertical_velocity",
    133: "specific_humidity",
    162051: "geopotential_at_surface",
    172: "land_sea_mask",
}

sl_inputs = {
    "2m_temperature": 167,
    "mean_sea_level_pressure": 151,
    "10m_v_component_of_wind": 166,
    "10m_u_component_of_wind": 165,
    "toa_incident_solar_radiation": 212,
}

pl_inputs = {
    "temperature": 130,
    "geopotential": 129,
    "u_component_of_wind": 131,
    "v_component_of_wind": 132,
    "vertical_velocity": 135,
    "specific_humidity": 133,
}

static_inputs = {
    "geopotential_at_surface": 162051,
    "land_sea_mask": "172",
}

levels = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]

time_dependent = {
    "toa_incident_solar_radiation": None,
    "year_progress_sin": None,
    "year_progress_cos": None,
    "day_progress_sin": None,
    "day_progress_cos": None,
}


def get_codes(variables: List[str], levels: List[int], time_levels: List[int]):
    """This defines a precise notion of input and output channels. Describing the
    input and output channels of graphcast requires a tuple of

    - The time level: t-1, t, or t+1
    - The ECWMF parameter ID
    - The pressure level if applicable

    For convenience and backwards compatibility, it is nice to have a string
    representation of this tuple (e.g. "t850" -> (id=130, level=850)).
    earth2mip.initial_conditions.cds has utilities for doing this, but does not
    include "history". please note there is no I/O in this code.
    """
    lookup_code = cds.keys_to_vals(CODE_TO_GRAPHCAST_NAME)
    output = []
    for v in sorted(variables):
        if v in time_dependent:
            for history in time_levels:
                output.append((history, v))
        elif v in static_inputs:
            output.append(v)
        elif v in lookup_code:
            code = lookup_code[v]
            if v in pl_inputs:
                for history in time_levels:
                    for level in levels:
                        output.append(
                            (history, cds.PressureLevelCode(code, level=level))
                        )
            else:
                for history in time_levels:
                    output.append((history, cds.SingleLevelCode(code)))
        else:
            raise NotImplementedError(v)
    return output


def get_state_codes(task_config: TaskConfig, time_level: int = 0):
    state_variables = [
        v for v in task_config.target_variables if v in task_config.input_variables
    ]
    return get_codes(
        state_variables, levels=task_config.pressure_levels, time_levels=[time_level]
    )


def get_data_for_code_scalar(code, scalar):
    match code:
        case _, cds.PressureLevelCode(id, level):
            arr = scalar[CODE_TO_GRAPHCAST_NAME[id]].sel(level=level).values
        case _, cds.SingleLevelCode(id):
            arr = scalar[CODE_TO_GRAPHCAST_NAME[id]].values
        case "land_sea_mask":
            arr = scalar[code].values
        case "geopotential_at_surface":
            arr = scalar[code].values
        case _, str(s):
            arr = scalar[s].values
    return arr


def get_codes_from_task_config(task_config: TaskConfig):
    x_codes = get_codes(
        task_config.input_variables,
        levels=task_config.pressure_levels,
        time_levels=[0, 1],
    )
    f_codes = get_codes(
        task_config.forcing_variables,
        levels=task_config.pressure_levels,
        time_levels=[2],
    )
    t_codes = get_codes(
        task_config.target_variables,
        levels=task_config.pressure_levels,
        time_levels=[0],
    )
    return x_codes + f_codes, t_codes


def torch_to_jax(x):
    return jax.dlpack.from_dlpack(torch.to_dlpack(x))


class NoXarrayGraphcast(graphcast.GraphCast):
    """A graphcast model that does not use xarray

    When initially developing this feature, the xarray logic was introducing
    NaNs that were difficult to track down.  For this reason, we wrap the core
    graphcast ML model which takes a single array of inputs with shape [nlat*nlon, batch, channels].

    Here is the original __call__ implementation:
    https://github.com/google-deepmind/graphcast/blob/858301cde5de5c728f8172f782dafba1ea07ac2e/graphcast/graphcast.py#L357

    """

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
    """
    This fucntion is mostly copied from
    https://github.com/google-deepmind/graphcast/tree/main

    License info:

    # Copyright 2023 DeepMind Technologies Limited.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #      http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS-IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    """
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
        target_scale: np.ndarray,
        task_config,
        # TODO move into grid
        lat: np.ndarray,
        lon: np.ndarray,
    ):
        in_codes, t_codes = get_codes_from_task_config(task_config)
        self.lon = lon
        self.lat = lat
        self.task_config = task_config
        self.forward = forward
        self._static_variables = static_variables
        self.mean = mean
        self.scale = scale
        self.target_scale = target_scale
        self.in_codes = in_codes
        self.target_codes = t_codes

        self.prog_levels = [
            [in_codes.index(k) for k in get_state_codes(task_config, t)]
            for t in range(2)
        ]

        self.in_channel_names = [str(c) for _, c in get_state_codes(task_config, 0)]

        # setup output names
        state_codes = get_state_codes(self.task_config, 0)
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

        timestamp = earth2mip.time.datetime_to_timestamp(time)
        tisr = toa_incident_solar_radiation_accumulated(timestamp, lat, lon)

        return self.set_forcing(x, "toa_incident_solar_radiation", t, tisr)

    def set_prognostic(self, array, t: int, data):
        index = self.prog_levels[t]
        return array.at[:, :, index].set(data)

    def get_prognostic(self, array, t: int):
        index = self.prog_levels[t]
        return array[:, :, index]

    def split_target(self, target):
        state_codes = get_state_codes(self.task_config, 0)
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
        d = self.forward(rng=rng, x=x) * self.target_scale
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


def _load_time_loop_from_description(
    package,
    checkpoint: str,
    resolution: float,
    nlevels,
    pretrained=True,
    device="cuda:0",
):
    def join(*args):
        return package.get(os.path.join(*args))

    checkpoint_path = join("params", model.checkpoint)
    # load checkpoint:
    with open(checkpoint_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
        task_config = ckpt.task_config
        run_forward = load_run_forward_from_checkpoint(
            ckpt, grid=GraphcastTimeLoop.grid
        )

    size = os.path.getsize(checkpoint_path)
    logging.info(f"Checkpoint Size in MB: {size / 1e6}")

    static_variables = get_static_data(package, model.resolution)

    # load stats
    with open(join("stats/diffs_stddev_by_level.nc"), "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open(join("stats/mean_by_level.nc"), "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open(join("stats/stddev_by_level.nc"), "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()

    # select needed channels from stats
    in_codes, t_codes = get_codes_from_task_config(task_config)
    mean = np.array(
        [get_data_for_code_scalar(code, mean_by_level) for code in in_codes]
    )
    scale = np.array(
        [get_data_for_code_scalar(code, stddev_by_level) for code in in_codes]
    )
    target_scale = np.array(
        [get_data_for_code_scalar(code, diffs_stddev_by_level) for code in t_codes]
    )
    return GraphcastTimeLoop(
        run_forward,
        static_variables,
        mean,
        scale,
        target_scale,
        task_config,
        lat=GraphcastTimeLoop.grid.lat,
        lon=GraphcastTimeLoop.grid.lon,
    )


# explicit graphcast versions
def load_time_loop(
    package,
    pretrained=True,
    device="cuda:0",
):
    return _load_time_loop_from_description(
        package=package,
        checkpoint="GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz",
        resolution=0.25,
        levels=37,
        pretrained=pretrained,
        device=device,
    )


def load_time_loop_small(
    package,
    pretrained=True,
    device="cuda:0",
):
    return _load_time_loop_from_description(
        package=package,
        checkpoint="GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
        resolution=1.0,
        levels=13,
        pretrained=pretrained,
        device=device,
    )


def load_time_loop_operational(
    package,
    pretrained=True,
    device="cuda:0",
):
    return _load_time_loop_from_description(
        package=package,
        checkpoint="GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz",
        resolution=0.25,
        levels=13,
        pretrained=pretrained,
        device=device,
    )
