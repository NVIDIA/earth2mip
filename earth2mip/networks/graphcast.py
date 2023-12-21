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
# TODO add license text from graphcast
import dataclasses
import functools
import os

from earth2mip.time_loop import TimeStepperLoop

__all__ = ["load_time_loop", "load_time_loop_operational", "load_time_loop_small"]


import datetime
import warnings

import haiku as hk
import jax
import jax.dlpack
import joblib
import numpy as np
import pandas as pd
import torch
import xarray
from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    xarray_jax,
)
from graphcast.data_utils import add_derived_vars
from graphcast.rollout import _get_next_inputs
from modulus.utils import zenith_angle

import earth2mip.grid
from earth2mip import time_loop
from earth2mip.initial_conditions import cds

# see ecwmf parameter table https://codes.ecmwf.int/grib/param-db/?&filter=grib1&table=128 # noqa
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


def jax_to_torch(x):
    return torch.from_dlpack(jax.dlpack.to_dlpack(x))


def torch_to_jax(x):
    # contiguous is important to avoid very mysterious errors with mixed up
    # channels. dlpack is not reliable with non-contiguous tensors
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))


class CachedGraphcast(graphcast.GraphCast):
    """GraphCast with cached graph structures"""

    def _maybe_init(self, sample_inputs):
        if self._initialized:
            return
        else:
            self._init_mesh_properties()
            self._init_grid_properties(
                grid_lat=sample_inputs.lat, grid_lon=sample_inputs.lon
            )

            if os.path.exists(".cache.pkl"):
                print("Loading cached graph structures from .cache.pkl")
                (
                    self._mesh2grid_graph_structure,
                    self._mesh_graph_structure,
                    self._grid2mesh_graph_structure,
                ) = joblib.load(".cache.pkl")
            else:
                self._grid2mesh_graph_structure = self._init_grid2mesh_graph()
                self._mesh_graph_structure = self._init_mesh_graph()
                self._mesh2grid_graph_structure = self._init_mesh2grid_graph()
                print("Saving graph structures to .cache.pkl")
                joblib.dump(
                    [
                        self._mesh2grid_graph_structure,
                        self._mesh_graph_structure,
                        self._grid2mesh_graph_structure,
                    ],
                    ".cache.pkl",
                )
            self._initialized = True


def get_channel_names(variables, pressure_levels):
    """Return e2mip style channel names like "z500" for a packed array

    The packed array contains ``variables`` and ``pressure levels``.
    """
    vars_3d = sorted(set(graphcast.ALL_ATMOSPHERIC_VARS) & set(variables))
    vars_surface = sorted(set(graphcast.TARGET_SURFACE_VARS) & set(variables))
    graphcast_name_to_code = {val: key for key, val in CODE_TO_GRAPHCAST_NAME.items()}

    pl_codes = [
        cds.PressureLevelCode(id=graphcast_name_to_code[v], level=level)
        for v in vars_3d
        for level in pressure_levels
    ]
    sl_codes = [cds.SingleLevelCode(id=graphcast_name_to_code[v]) for v in vars_surface]
    all_codes = pl_codes + sl_codes
    names = [str(c) for c in all_codes]
    return names


def _get_array_module(x):
    """
    Used for device agnostic code, following this pattern:
    https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code
    """
    if isinstance(x, jax.Array):
        return jax.numpy
    elif isinstance(x, np.ndarray):
        return np
    else:
        raise ValueError(f"Unknown array type {type(x)}")


def _tisr(t, lat, lon):
    """compute tisr with jax or numpy and without unnecessary warnings

    Should be upstreamed to modulus
    """
    xp = _get_array_module(t)
    try:
        old_np = zenith_angle.np
        zenith_angle.np = xp
        # catch this warning
        # /usr/local/lib/python3.10/dist-packages/modulus/utils/zenith_angle.py:276:
        # RuntimeWarning: invalid value encountered in arccos
        # hc = np.arccos(-A / B)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return zenith_angle.toa_incident_solar_radiation_accumulated(t, lat, lon)
    finally:
        zenith_angle.np = old_np


def _get_tisr(seconds_since_epoch, lat, lon):
    t = seconds_since_epoch[..., None, None]
    lat = lat[:, None]
    lon = lon[None, :]
    tisr = _tisr(t, lat, lon)
    return xarray_jax.Variable(["batch", "time", "lat", "lon"], tisr)


def get_forcings(time, lat, lon):
    """
    Args:
        time: (batch, time) shaped array
        lat: (lat,) shaped array
        lon: (lon,) shaped array
    Returns:
        forcings: Dataset, maximum dims are (batch, time, lat, lon)

    """

    forcings = xarray.Dataset()
    # need to cast to datetime64[ns] to avoid an xarray warning
    forcings["datetime"] = (["batch", "time"], time.astype("datetime64[ns]"))
    forcings["lon"] = (["lon"], lon)
    forcings["lat"] = (["lat"], lat)
    forcings = forcings.set_coords(["datetime", "lon", "lat"])
    seconds_since_epoch = (
        forcings.coords["datetime"].data.astype("datetime64[s]").astype(np.int64)
    )

    # unfortunately, this needs to run on the CPU since it uses the datetime
    # information
    add_derived_vars(forcings)
    forcings = forcings.drop_vars("datetime")
    forcings = forcings.transpose("batch", "time", "lat", "lon")

    # put data on same device as lat
    if isinstance(lat, jax.Array):
        forcings = jax.tree_map(
            lambda x: jax.device_put(x, device=lat.device()), forcings
        )
        seconds_since_epoch = jax.device_put(seconds_since_epoch, device=lat.device())

    forcings["toa_incident_solar_radiation"] = _get_tisr(seconds_since_epoch, lat, lon)

    return forcings


def torch_device_to_jax(device: torch.device) -> jax.Device:
    x = torch.ones([], device=device)
    return torch_to_jax(x).device()


class GraphcastStepper(time_loop.TimeStepper):
    """

    Some information about the input and output state of graphcast::

        # Inputs
        eval inputs:
        xarray.Dataset {
        dimensions:
                batch = 1 ;
                time = 2 ;
                lat = 721 ;
                lon = 1440 ;
                level = 37 ;

        variables:
                float32 2m_temperature(batch, time, lat, lon) ;
                float32 mean_sea_level_pressure(batch, time, lat, lon) ;
                float32 10m_v_component_of_wind(batch, time, lat, lon) ;
                float32 10m_u_component_of_wind(batch, time, lat, lon) ;
                float32 total_precipitation_6hr(batch, time, lat, lon) ;
                float32 temperature(batch, time, level, lat, lon) ;
                float32 geopotential(batch, time, level, lat, lon) ;
                float32 u_component_of_wind(batch, time, level, lat, lon) ;
                float32 v_component_of_wind(batch, time, level, lat, lon) ;
                float32 vertical_velocity(batch, time, level, lat, lon) ;
                float32 specific_humidity(batch, time, level, lat, lon) ;
                float32 toa_incident_solar_radiation(batch, time, lat, lon) ;
                float32 year_progress_sin(batch, time) ;
                float32 year_progress_cos(batch, time) ;
                float32 day_progress_sin(batch, time, lon) ;
                float32 day_progress_cos(batch, time, lon) ;
                float32 geopotential_at_surface(lat, lon) ;
                float32 land_sea_mask(lat, lon) ;
                float32 lon(lon) ;
                        lon:long_name = longitude ;
                        lon:units = degrees_east ;
                float32 lat(lat) ;
                        lat:long_name = latitude ;
                        lat:units = degrees_north ;
                int32 level(level) ;
                timedelta64[ns] time(time) ;

        // global attributes:
        }
        <xarray.DataArray 'time' (time: 2)>
        array([-21600000000000,               0], dtype='timedelta64[ns]')
        Coordinates:
        * time     (time) timedelta64[ns] -1 days +18:00:00 00:00:00


        # forcings
        xarray.Dataset {
        dimensions:
                batch = 1 ;
                time = 1 ;
                lat = 721 ;
                lon = 1440 ;

        variables:
                float32 toa_incident_solar_radiation(batch, time, lat, lon) ;
                float32 year_progress_sin(batch, time) ;
                float32 year_progress_cos(batch, time) ;
                float32 day_progress_sin(batch, time, lon) ;
                float32 day_progress_cos(batch, time, lon) ;
                float32 lon(lon) ;
                        lon:long_name = longitude ;
                        lon:units = degrees_east ;
                float32 lat(lat) ;
                        lat:long_name = latitude ;
                        lat:units = degrees_north ;
                timedelta64[ns] time(time) ;

        // global attributes:
        }<xarray.DataArray 'time' (time: 1)>
        array([21600000000000], dtype='timedelta64[ns]')
        Coordinates:
        * time     (time) timedelta64[ns] 06:00:00


        # Outputs

        predictions:
        xarray.Dataset {
        dimensions:
                time = 1 ;
                batch = 1 ;
                lat = 721 ;
                lon = 1440 ;
                level = 37 ;

        variables:
                float32 10m_u_component_of_wind(time, batch, lat, lon) ;
                float32 10m_v_component_of_wind(time, batch, lat, lon) ;
                float32 2m_temperature(time, batch, lat, lon) ;
                float32 geopotential(time, batch, level, lat, lon) ;
                float32 mean_sea_level_pressure(time, batch, lat, lon) ;
                float32 specific_humidity(time, batch, level, lat, lon) ;
                float32 temperature(time, batch, level, lat, lon) ;
                float32 total_precipitation_6hr(time, batch, lat, lon) ;
                float32 u_component_of_wind(time, batch, level, lat, lon) ;
                float32 v_component_of_wind(time, batch, level, lat, lon) ;
                float32 vertical_velocity(time, batch, level, lat, lon) ;
                float32 lon(lon) ;
                        lon:long_name = longitude ;
                        lon:units = degrees_east ;
                float32 lat(lat) ;
                        lat:long_name = latitude ;
                        lat:units = degrees_north ;
                int32 level(level) ;
                timedelta64[ns] time(time) ;

        // global attributes:
        }
        <xarray.DataArray 'time' (time: 1)>
        array([21600000000000], dtype='timedelta64[ns]')
        Coordinates:
        * time     (time) timedelta64[ns] 06:00:00
    """

    def __init__(
        self,
        run_forward,
        eval_inputs,
        eval_targets,
        eval_forcings,
        task_config,
        device=None,
    ):
        self.run_forward = run_forward
        self.eval_inputs = eval_inputs
        self.eval_targets = eval_targets
        self.eval_forcings = eval_forcings
        self.task_config = task_config
        lat = eval_inputs.lat.values
        lon = eval_inputs.lon.values
        self._grid = earth2mip.grid.LatLonGrid(lat.tolist(), lon.tolist())
        self._history_time_step = pd.Timedelta("6h")
        self._n_history_levels = (
            pd.Timedelta(task_config.input_duration) // self._history_time_step
        )
        self._device = device or torch.cuda.current_device()
        self._jax_device = torch_device_to_jax(self._device)
        self.lat = jax.device_put(lat, device=self._jax_device)
        self.lon = jax.device_put(lon, device=self._jax_device)

    @property
    def input_info(self) -> time_loop.GeoTensorInfo:
        return time_loop.GeoTensorInfo(
            channel_names=self._get_in_channel_names(),
            grid=self._grid,
            n_history_levels=self._n_history_levels,
            history_time_step=self._history_time_step,
        )

    @property
    def output_info(self) -> time_loop.GeoTensorInfo:
        return time_loop.GeoTensorInfo(
            channel_names=self._out_channel_names,
            grid=self._grid,
            n_history_levels=self._n_history_levels,
            history_time_step=self._history_time_step,
        )

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return self._device

    @property
    def time_step(self) -> datetime.timedelta:
        return datetime.timedelta(hours=6)

    def _assert_inputs_are_on_the_correct_device(self, inputs):
        first_input = self.task_config.input_variables[0]
        data = xarray_jax.unwrap_data(inputs[first_input])
        assert data.device() == self._jax_device

    def initialize(self, x: torch.Tensor, time: datetime.datetime):
        x_jax = torch_to_jax(x)
        time = pd.Timestamp(time)
        dt = pd.Timedelta(self.time_step)
        inputs = self._get_inputs(x_jax, time, dt)
        self._assert_inputs_are_on_the_correct_device(inputs)
        rng = jax.random.PRNGKey(0)
        state = (time, inputs, rng)
        return state

    def step(self, state):
        time, inputs, rng = state
        time, inputs, predictions_xr, rng = self._step(time, inputs, rng)
        array = self._pack(predictions_xr)
        tensor = jax_to_torch(array)
        new_state = (time, inputs, rng)
        assert tensor.shape[1] == 1, "targets should only contain 1 time level"  # noqa
        return new_state, tensor[:, 0]

    def _step(self, time, inputs, rng):
        rng, this_rng = jax.random.split(rng)

        forcings_template = self.eval_forcings.isel(time=slice(0, 1))
        target_template = self.eval_targets.isel(time=slice(0, 1))

        # get forcings
        forcing_time = time + forcings_template.time.values
        # add batch dim
        forcing_time = forcing_time[None]
        forcings = get_forcings(forcing_time, self.lat, self.lon)
        forcings = forcings.assign_coords(
            time=self.eval_forcings.time.isel(time=slice(0, 1))
        )
        del forcings["year_progress"]
        del forcings["day_progress"]

        predictions = self.run_forward(
            rng=this_rng,
            inputs=inputs,
            targets_template=target_template,
            forcings=forcings,
        )
        time = time + np.timedelta64(6, "h")
        next_frame = xarray.merge([predictions, forcings])
        return time, _get_next_inputs(inputs, next_frame), predictions, rng

    def _get_num_channels_x(self):
        vars_3d = sorted(
            set(graphcast.ALL_ATMOSPHERIC_VARS) & set(self.task_config.input_variables)
        )
        vars_surface = sorted(
            set(graphcast.TARGET_SURFACE_VARS) & set(self.task_config.input_variables)
        )
        return len(vars_3d) * len(self.task_config.pressure_levels) + len(vars_surface)

    def _get_in_channel_names(self) -> list[str]:
        return get_channel_names(
            self.task_config.input_variables, self.task_config.pressure_levels
        )

    @property
    def _out_channel_names(self) -> list[str]:
        return get_channel_names(
            self.task_config.target_variables, self.task_config.pressure_levels
        )

    @staticmethod
    def _pack(ds: xarray.Dataset) -> np.ndarray:
        """Stack a dataset into a single array

        The inverse operation to get_inputs. an xarray dataset is stacked, first
        the 3d variables, and then the surface variables. Forcings and static
        variables are ignored. Variable names are sorted before stacking to
        ensure deterministic order.

        Returns:
            (batch, time, channel, lat, lon) shaped array.

        """
        vars_3d = sorted(set(graphcast.ALL_ATMOSPHERIC_VARS) & set(ds))
        vars_surface = sorted(set(graphcast.TARGET_SURFACE_VARS) & set(ds))

        pl = [
            xarray_jax.unwrap(
                ds[v].transpose("batch", "time", "level", "lat", "lon").data
            )
            for v in vars_3d
        ]
        sl = [xarray_jax.unwrap(ds[v].data)[:, :, np.newaxis] for v in vars_surface]

        return jax.numpy.concatenate(pl + sl, axis=2)

    def _get_inputs(self, x, time, dt):
        """get xarray inputs from stacked array x

        Args:
            x: (batch, time, channel, lat, lon) shaped array
                packed along the channel dimension. The order is 3d variables,
                then surface.
            time: the time of x[:, -1] (np.timedelta64)
            dt: the time difference along the time dimension
        Returns:
            xarray.Dataset like eval_inputs. Forcings are computed from time, lat, lon.
        """
        b, t, _, nx, ny = x.shape
        levels = self.task_config.pressure_levels
        vars_3d = sorted(
            set(graphcast.ALL_ATMOSPHERIC_VARS) & set(self.task_config.input_variables)
        )
        vars_surface = sorted(
            set(graphcast.TARGET_SURFACE_VARS) & set(self.task_config.input_variables)
        )
        vars_static = sorted(
            set(graphcast.STATIC_VARS) & set(self.task_config.input_variables)
        )

        n3d = len(levels) * len(vars_3d)
        pl = x[:, :, :n3d]

        n2d = len(vars_surface)
        sl = x[:, :, n3d : n3d + n2d]

        assert n2d + n3d == x.shape[2]  # noqa
        inputs = xarray.Dataset()
        time_offset = np.arange(-t + 1, 1, 1) * dt
        time_offset = time_offset.astype("timedelta64[ns]")
        assert time_offset.shape == (t,)  # noqa
        inputs["time"] = (["time"], time_offset)
        inputs["level"] = (["level"], np.array(levels))

        pl = pl.reshape(b, t, len(vars_3d), len(levels), nx, ny)
        for i, var in enumerate(vars_3d):
            inputs[var] = xarray_jax.Variable(
                ["batch", "time", "level", "lat", "lon"], pl[:, :, i]
            )

        for i, var in enumerate(vars_surface):
            inputs[var] = xarray_jax.Variable(
                ["batch", "time", "lat", "lon"], sl[:, :, i]
            )

        for var in vars_static:
            inputs[var] = xarray_jax.Variable(
                dims=self.eval_inputs[var].dims,
                data=jax.device_put(
                    self.eval_inputs[var].data, device=self._jax_device
                ),
            )

        forcings = get_forcings(
            time + time_offset[None],
            self.eval_inputs.lat.values,
            self.eval_inputs.lon.values,
        )
        del forcings["year_progress"]
        del forcings["day_progress"]
        inputs.update(forcings)
        inputs = inputs.set_coords(["time", "level", "lat", "lon"])
        return inputs


def load_stepper(
    checkpoint_path: str,
    dataset_filename: str,
    stats_dir: str,
    device: torch.device = None,
):
    # load checkpoint:
    with open(checkpoint_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
        params = ckpt.params
        state = {}

    model_config = ckpt.model_config
    task_config = ckpt.task_config
    print("Model description:\n", ckpt.description, "\n")
    print("Model license:\n", ckpt.license, "\n")

    # load dataset
    example_batch = xarray.open_dataset(dataset_filename)
    # 2 for input, >=1 for targets
    assert example_batch.dims["time"] >= 3  # noqa

    # get eval data
    eval_steps = 1
    (
        eval_inputs,
        eval_targets,
        eval_forcings,
    ) = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{eval_steps*6}h"),
        **dataclasses.asdict(task_config),
    )

    print("All Examples:  ", example_batch.dims.mapping)
    print("Eval Inputs:   ", eval_inputs.dims.mapping)
    print("Eval Targets:  ", eval_targets.dims.mapping)
    print("Eval Forcings: ", eval_forcings.dims.mapping)

    # run autoregression
    assert model_config.resolution in (0, 360.0 / eval_inputs.sizes["lon"]), (  # noqa
        "Model resolution doesn't match the data resolution. You likely want to "
        "re-filter the dataset list, and download the correct data."
    )  # noqa

    print("Inputs:  ", eval_inputs.dims.mapping)
    print("Targets: ", eval_targets.dims.mapping)
    print("Forcings:", eval_forcings.dims.mapping)

    # load stats
    with open(os.path.join(stats_dir, "diffs_stddev_by_level.nc"), "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open(os.path.join(stats_dir, "mean_by_level.nc"), "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open(os.path.join(stats_dir, "stddev_by_level.nc"), "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()

    # jit the stuff
    # @title Build jitted functions, and possibly initialize random weights
    def construct_wrapped_graphcast(
        model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig
    ):
        """Constructs and wraps the GraphCast Predictor."""
        # Deeper one-step predictor.
        predictor = CachedGraphcast(model_config, task_config)

        # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
        # from/to float32 to/from BFloat16
        predictor = casting.Bfloat16Cast(predictor)

        # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
        # BFloat16 happens after applying normalization to the inputs/targets.
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )

        # Wraps everything so the one-step model can produce trajectories.
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

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

    run_forward_jitted = drop_state(
        with_params(jax.jit(with_configs(run_forward.apply)))
    )
    stepper = GraphcastStepper(
        run_forward_jitted,
        eval_inputs,
        eval_targets,
        eval_forcings,
        task_config,
        device=device,
    )
    return stepper


def _load_time_loop_from_description(
    package,
    checkpoint_path: str,
    dataset_path: str,
    device="cuda:0",
) -> TimeStepperLoop:
    checkpoint = package.get(os.path.join("params", checkpoint_path))
    dataset_path = package.get(os.path.join("dataset", dataset_path))
    stats_dir = package.get("stats", recursive=True)
    stepper = load_stepper(checkpoint, dataset_path, stats_dir, device=device)
    return TimeStepperLoop(stepper)


# explicit graphcast versions
def load_time_loop(
    package,
    pretrained=True,
    device="cuda:0",
) -> TimeStepperLoop:
    return _load_time_loop_from_description(
        package=package,
        checkpoint_path="GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz",  # noqa
        dataset_path="source-era5_date-2022-01-01_res-0.25_levels-37_steps-04.nc",
        device=device,
    )


def load_time_loop_small(
    package,
    pretrained=True,
    device="cuda:0",
) -> TimeStepperLoop:
    return _load_time_loop_from_description(
        package=package,
        checkpoint_path="GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",  # noqa
        dataset_path="source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc",
        device=device,
    )


def load_time_loop_operational(
    package,
    pretrained=True,
    device="cuda:0",
) -> TimeStepperLoop:
    return _load_time_loop_from_description(
        package=package,
        checkpoint_path="GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz",  # noqa
        dataset_path="source-era5_date-2022-01-01_res-0.25_levels-13_steps-04.nc",
        device=device,
    )
