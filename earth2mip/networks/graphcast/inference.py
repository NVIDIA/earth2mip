# %%
# @title Imports
import os
import numpy as np
import dataclasses
from earth2mip import schema
import datetime
import functools
import torch
import math
import re
from typing import Optional

import cartopy.crs as ccrs
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from earth2mip.networks.graphcast import rollout, channels
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray
import jax
from earth2mip.networks.graphcast.rollout import _get_next_inputs

rng = jax.random.PRNGKey(0)


def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))


# @title Plotting functions


def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> xarray.Dataset:
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if (
        max_steps is not None
        and "time" in data.sizes
        and max_steps < data.sizes["time"]
    ):
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data


def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (
        data,
        matplotlib.colors.Normalize(vmin, vmax),
        ("RdBu_r" if center is not None else "viridis"),
    )


def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols, plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    images = []
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(
            plot_data.isel(time=0, missing_dims="ignore"),
            norm=norm,
            origin="lower",
            cmap=cmap,
        )
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            cmap=cmap,
            extend=("both" if robust else "neither"),
        )
        images.append(im)

    def update(frame):
        if "time" in first_data.dims:
            td = datetime.timedelta(
                microseconds=first_data["time"][frame].item() / 1000
            )
            figure.suptitle(f"{fig_title}, {td}", fontsize=16)
        else:
            figure.suptitle(fig_title, fontsize=16)
        for im, (plot_data, norm, cmap) in zip(images, data.values()):
            im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

    ani = animation.FuncAnimation(
        fig=figure, func=update, frames=max_steps, interval=250
    )
    plt.close(figure.number)
    return HTML(ani.to_jshtml())


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


class Graphcast:

    grid = schema.Grid.grid_721x1440
    n_history_levels: int = 2
    history_time_step: datetime.timedelta = datetime.timedelta(hours=6)
    time_step: datetime.timedelta = datetime.timedelta(hours=6)
    dtype: torch.dtype = torch.float32

    def __init__(self, package, device):
        model_name = "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"

        self.device = device

        def join(*args):
            return package.get(os.path.join(*args))

        checkpoint_path = join("params", model_name)
        # load checkpoint:
        with open(checkpoint_path, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
            params = ckpt.params
            state = {}

        model_config = ckpt.model_config
        task_config = ckpt.task_config
        self.task_config = task_config
        print("Model description:\n", ckpt.description, "\n")
        print("Model license:\n", ckpt.license, "\n")

        # load stats
        with open(join("stats/diffs_stddev_by_level.nc"), "rb") as f:
            diffs_stddev_by_level = xarray.load_dataset(f).compute()
        with open(join("stats/mean_by_level.nc"), "rb") as f:
            mean_by_level = xarray.load_dataset(f).compute()
        with open(join("stats/stddev_by_level.nc"), "rb") as f:
            stddev_by_level = xarray.load_dataset(f).compute()

        # static data
        static_path = package.get(
            "dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc"
        )
        with xarray.open_dataset(static_path) as ds:
            self.static_data = ds[["geopotential_at_surface", "land_sea_mask"]].load()

        # jit the stuff
        # @title Build jitted functions, and possibly initialize random weights
        def construct_wrapped_graphcast(
            model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig
        ):
            """Constructs and wraps the GraphCast Predictor."""
            # Deeper one-step predictor.
            predictor = graphcast.GraphCast(model_config, task_config)

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

            # # Wraps everything so the one-step model can produce trajectories.
            # predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
            return predictor

        @hk.transform_with_state
        def run_forward(model_config, task_config, inputs, targets_template, forcings):
            predictor = construct_wrapped_graphcast(model_config, task_config)
            return predictor(
                inputs, targets_template=targets_template, forcings=forcings
            )

        # Jax doesn't seem to like passing configs as args through the jit. Passing it
        # in via partial (instead of capture by closure) forces jax to invalidate the
        # jit cache if you change configs.
        def with_configs(fn):
            return functools.partial(
                fn, model_config=model_config, task_config=task_config
            )

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

        self.run_forward_jitted = drop_state(
            with_params(jax.jit(with_configs(run_forward.apply)))
        )

    @property
    def codes(self):
        return channels.get_graphcast_codes(channels.levels, precip=True)

    @property
    def in_channel_names(self):
        return [str(c) for c in self.codes]

    @property
    def out_channel_names(self):
        return self.in_channel_names

    def __call__(self, time, x):
        """

        inputs::

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

        targets::

            xarray.Dataset {
            dimensions:
                    batch = 1 ;
                    time = 12 ;
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

        forcings::

            xarray.Dataset {
            dimensions:
                    batch = 1 ;
                    time = 12 ;
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
            }

        """
        # graphcast uses -90 to 90 for lat
        x_rev = x[:, :, :, ::-1]
        d = channels.unpack_all(x_rev, self.codes)

        # states up to t
        d = channels.assign_grid_coords(d, self.grid)
        d.coords["time"] = (
            np.arange(self.n_history_levels) - self.n_history_levels + 1
        ) * self.history_time_step
        channels.add_dynamic_vars(d, time)
        d = d.merge(self.static_data)

        # "targets" template. This this is basically a schema of the state at t
        # + 1. Not sure why it's needed.
        target_vars = [
            "2m_temperature",
            "mean_sea_level_pressure",
            "10m_v_component_of_wind",
            "10m_u_component_of_wind",
            "total_precipitation_6hr",
            "temperature",
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "specific_humidity",
        ]
        target = (
            d[target_vars].isel(time=[0]).assign(time=[self.history_time_step]) * np.nan
        )

        while True:
            final_time = d.isel(time=slice(0, 1))
            out = channels.pack(final_time, self.codes)
            # need to reverse lat
            yield time, out[:, 0, :, ::-1]
            # forcings at t + 1
            forcings = xarray.Dataset()
            forcings.coords["time"] = xarray.Variable(
                ["time"], [self.history_time_step]
            )
            forcings = channels.assign_grid_coords(forcings, self.grid)
            channels.add_dynamic_vars(forcings, time)

            next_inputs = self.run_forward_jitted(
                rng=rng, inputs=d, targets_template=target, forcings=forcings
            )

            next_inputs = next_inputs.merge(forcings)
            d = _get_next_inputs(d, next_inputs)
            time += self.time_step

        return

        # TODO restore these tests

        def assert_schema_same(x, y):
            assert set(x) == set(y), set(x).symmetric_difference(set(y))
            for v in x:
                assert x[v].dims == y[v].dims
            xarray.testing.assert_equal(x.coords.to_dataset(), y.coords.to_dataset())

        assert_schema_same(d, eval_inputs)
        assert_schema_same(forcings, eval_forcings.isel(time=[0]))
        assert_schema_same(target, eval_targets.isel(time=[0]))


def main():
    root = "/lustre/fsw/sw_earth2_ml/graphcast/"
    model = Graphcast(root)

    # load dataset
    dataset_filename = join(
        root, "dataset", "source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc"
    )
    with open(dataset_filename, "rb") as f:
        example_batch = xarray.load_dataset(f).compute()

    assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

    # get eval data
    eval_steps = 12
    (
        eval_inputs,
        eval_targets,
        eval_forcings,
    ) = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{eval_steps*6}h"),
        **dataclasses.asdict(model.task_config),
    )

    eval_inputs.info()
    eval_targets.info()
    eval_forcings.info()
    predictions = model(eval_inputs, eval_targets, eval_forcings)
    print(predictions)


if __name__ == "__main__":
    main()
