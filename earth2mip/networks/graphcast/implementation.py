"""
# TODO add license text from graphcast
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
# @title Imports
import os
import dataclasses
import functools

import jax.dlpack
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import xarray_jax
from graphcast.rollout import _get_next_inputs
from graphcast.data_utils import add_derived_vars
import haiku as hk
import jax
import numpy as np
import xarray
import joblib
import warnings

from modulus.utils.zenith_angle import toa_incident_solar_radiation_accumulated
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


def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))


# @title Plotting functions
def data_valid_for_model(
    file_name: str,
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
):

    file_parts = parse_file_parts(file_name.removesuffix(".nc"))

    return (
        model_config.resolution in (0, float(file_parts["res"]))
        and len(task_config.pressure_levels) == int(file_parts["levels"])
        and (
            (
                "total_precipitation_6hr" in task_config.input_variables
                and file_parts["source"] in ("era5", "fake")
            )
            or (
                "total_precipitation_6hr" not in task_config.input_variables
                and file_parts["source"] in ("hres", "fake")
            )
        )
    )


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


class GraphcastStepper:
    def __init__(
        self, run_forward, eval_inputs, eval_targets, eval_forcings, task_config
    ):
        self.run_forward = run_forward
        self.eval_inputs = eval_inputs
        self.eval_targets = eval_targets
        self.eval_forcings = eval_forcings
        self.task_config = task_config
        self.lat = eval_inputs.lat.values
        self.lon = eval_inputs.lon.values

    def get_forcings(self, time, lat, lon):
        """
        Args:
            time: (batch, time) shaped array
            lat: (lat,) shaped array
            lon: (lon,) shaped array
        Returns:
            forcings: Dataset, maximum dims are (batch, time, lat, lon)

        """

        forcings = xarray.Dataset()
        forcings["datetime"] = (["batch", "time"], time)
        forcings["lon"] = (["lon"], self.lon)
        forcings["lat"] = (["lat"], self.lat)
        forcings = forcings.set_coords(["datetime", "lon", "lat"])
        seconds_since_epoch = (
            forcings.coords["datetime"].data.astype("datetime64[s]").astype(np.int64)
        )
        t = seconds_since_epoch[..., None, None]
        lat = lat[:, None]
        lon = lon[None, :]

        # catch this warning
        # /usr/local/lib/python3.10/dist-packages/modulus/utils/zenith_angle.py:276:
        # RuntimeWarning: invalid value encountered in arccos
        # hc = np.arccos(-A / B)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tisr = toa_incident_solar_radiation_accumulated(t, lat, lon)

        forcings["toa_incident_solar_radiation"] = (
            ["batch", "time", "lat", "lon"],
            tisr,
        )
        add_derived_vars(forcings)
        forcings = forcings.drop_vars("datetime")
        return forcings.transpose("batch", "time", "lat", "lon")

    def step(self, time, inputs, rng):
        rng, this_rng = jax.random.split(rng)

        forcings_template = self.eval_forcings.isel(time=slice(0, 1))
        target_template = self.eval_targets.isel(time=slice(0, 1))

        # get forcings
        forcing_time = time + forcings_template.time.values
        # add batch dim
        forcing_time = forcing_time[None]
        forcings = self.get_forcings(forcing_time, self.lat, self.lon)
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

    def get_num_channels_x(self):
        vars_3d = sorted(
            set(graphcast.ALL_ATMOSPHERIC_VARS) & set(self.task_config.input_variables)
        )
        vars_surface = sorted(
            set(graphcast.TARGET_SURFACE_VARS) & set(self.task_config.input_variables)
        )
        return len(vars_3d) * len(self.task_config.pressure_levels) + len(vars_surface)

    def get_in_channel_names(self) -> list[str]:
        return get_channel_names(
            self.task_config.input_variables, self.task_config.pressure_levels
        )

    @property
    def out_channel_names(self) -> list[str]:
        return get_channel_names(
            self.task_config.target_variables, self.task_config.pressure_levels
        )

    @staticmethod
    def pack(ds: xarray.Dataset) -> np.ndarray:
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

    def get_inputs(self, x, time, dt):
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

        assert n2d + n3d == x.shape[2]
        inputs = xarray.Dataset()
        time_offset = np.arange(-t + 1, 1, 1) * dt
        assert time_offset.shape == (t,)
        inputs["time"] = (["time"], time_offset)
        inputs["level"] = (["level"], np.array(levels))

        pl = pl.reshape(b, t, len(vars_3d), len(levels), nx, ny)
        for i, var in enumerate(vars_3d):
            inputs[var] = (["batch", "time", "level", "lat", "lon"], pl[:, :, i])

        for i, var in enumerate(vars_surface):
            inputs[var] = (["batch", "time", "lat", "lon"], sl[:, :, i])

        for var in vars_static:
            inputs[var] = self.eval_inputs[var]

        forcings = self.get_forcings(
            time + time_offset[None],
            self.eval_inputs.lat.values,
            self.eval_inputs.lon.values,
        )
        del forcings["year_progress"]
        del forcings["day_progress"]
        inputs.update(forcings)
        inputs = inputs.set_coords(["time", "level", "lat", "lon"])

        return inputs


def load_graphcast(
    checkpoint_path: str,
    dataset_filename: str,
    stats_dir: str,
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
    assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

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
    assert model_config.resolution in (0, 360.0 / eval_inputs.sizes["lon"]), (
        "Model resolution doesn't match the data resolution. You likely want to "
        "re-filter the dataset list, and download the correct data."
    )

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
        run_forward_jitted, eval_inputs, eval_targets, eval_forcings, task_config
    )
    return stepper
