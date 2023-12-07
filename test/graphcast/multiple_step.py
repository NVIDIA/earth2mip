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
# TODO add graphcast license
"""
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
import dataclasses
import functools
import os

import haiku as hk
import jax
import jax.dlpack
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray
from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
)

from earth2mip.networks.graphcast import GraphcastStepper


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


def main():
    model_name = "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"  # noqa
    root = "/lustre/fsw/sw_earth2_ml/graphcast/"
    root = "/home/nbrenowitz/mnt/selene/fsw/sw_earth2_ml/graphcast/"
    root = ".tmp"
    checkpoint_path = os.path.join(root, "params", model_name)

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
    dataset_filename = os.path.join(
        root, "dataset", "source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc"
    )
    if not data_valid_for_model(
        os.path.basename(dataset_filename), model_config, task_config
    ):
        raise ValueError(
            f"Invalid dataset file {os.path.basename(dataset_filename)}, rerun the cell above and choose a valid dataset file."  # noqa
        )

    example_batch = xarray.open_dataset(dataset_filename)
    assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

    # get eval data
    eval_steps = 10
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
    with open(os.path.join(root, "stats/diffs_stddev_by_level.nc"), "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open(os.path.join(root, "stats/mean_by_level.nc"), "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open(os.path.join(root, "stats/stddev_by_level.nc"), "rb") as f:
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

    init_jitted = jax.jit(with_configs(run_forward.init))

    if params is None:
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=eval_inputs,
            targets_template=eval_targets,
            forcings=eval_forcings,
        )

    run_forward_jitted = drop_state(
        with_params(jax.jit(with_configs(run_forward.apply)))
    )

    # time is actually the penultimate time step
    time = example_batch.datetime[0, -(eval_steps + 1)].values
    dt = pd.Timedelta("6h")

    stepper = GraphcastStepper(
        run_forward_jitted, eval_inputs, eval_targets, eval_forcings, task_config
    )
    import torch.utils.tensorboard as tb

    writer = tb.SummaryWriter("runs")

    # test get_inputs
    shape = (1, 2, stepper._get_num_channels_x(), 721, 1440)
    x = np.arange(np.prod(shape)).reshape(shape)
    state = stepper._get_inputs(x, time, dt)
    x_round_trip = stepper._pack(state)
    np.testing.assert_array_equal(x, x_round_trip)

    # test on real inputs
    shape = (1, 2, stepper._get_num_channels_x(), 721, 1440)
    x = stepper._pack(eval_inputs)
    state = stepper._get_inputs(x, time, dt)
    tisr_name = "toa_incident_solar_radiation"
    xarray.testing.assert_equal(state.drop(tisr_name), eval_inputs.drop(tisr_name))
    for t in range(2):
        fig, (a, b, c) = plt.subplots(3, 1, figsize=(10, 10))

        truth = eval_inputs.toa_incident_solar_radiation[0, t]
        ours = state.toa_incident_solar_radiation[0, t]

        truth.plot.imshow(ax=a)
        ours.plot.imshow(ax=b)
        (truth - ours).plot.imshow(ax=c)
        a.set_title("truth")
        b.set_title("ours")
        c.set_title("diff")
        writer.add_figure("input_tisr", fig, t)
    # TODO this is a large tolerance (though only .1% in terms of peak amplitude
    # of TISR)
    xarray.testing.assert_allclose(state[tisr_name], eval_inputs[tisr_name], atol=3500)

    # TODO test get_forcings

    # TEST channel_names
    names = stepper._get_in_channel_names()
    assert len(names) == stepper._get_num_channels_x()
    print(names)

    # run simulation
    state = (time, eval_inputs, jax.random.PRNGKey(0))
    states = []
    for t in range(5):
        print(time)
        state, output = stepper.step(state)
        time, inputs, _ = state

        assert not np.any(np.isnan(inputs)).to_array().any()
        plotme = output[0, stepper.output_info.channel_names.index("q925")]
        plt.imshow(plotme.cpu())
        writer.add_figure("q925", plt.gcf(), t)
        states.append(inputs.isel(time=-1))

    predictions = xarray.concat(states, dim="time")

    # Compare against reference
    reference = xarray.open_dataset("predictions.nc")
    for var in predictions:
        predictions[var].data = predictions[var].values
    predictions = predictions.drop("time")
    reference = reference.isel(time=slice(0, predictions.sizes["time"]))

    level = 500
    field = "u_component_of_wind"
    time = 0

    fig, (a, b, c) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)
    p = predictions[field].sel(level=level)[0, time]
    t = reference[field].sel(level=level)[time, 0]
    diff = t - p
    p.plot.imshow(ax=a)
    a.set_title("predictions")
    t.plot.imshow(ax=b)
    b.set_title("reference")
    diff.plot.imshow(ax=c)
    c.set_title("diff")

    for ax in [a, b, c]:
        ax.grid()

    writer.add_figure(f"difference/{time}/{field}/{level}", fig, time)

    # ensure that difference wrt reference.  This looks like a large tolerance
    # but does detect shifts in time of 6 hours the image clearly shows the
    # difference is small
    t = reference.isel(time=slice(0, 2))
    num = predictions.isel(time=slice(0, 2)) - t
    nn = np.abs(num).mean(["batch", "time", "lat", "lon"])
    denom = np.abs(t).mean(["batch", "time", "lat", "lon"])
    lims = nn < denom * 0.01
    fraction_close = lims.mean().to_array().mean()
    assert fraction_close > 0.8


if __name__ == "__main__":
    main()
