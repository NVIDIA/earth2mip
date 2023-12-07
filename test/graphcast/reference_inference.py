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
# @title Imports
import dataclasses
import datetime
import functools
import math
import os
from typing import Optional

import haiku as hk
import jax
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray
from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
    xarray_jax,
    xarray_tree,
)
from IPython.display import HTML
from matplotlib import animation


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
    model_name = "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
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
            f"Invalid dataset file {os.path.basename(dataset_filename)}, rerun the cell above and choose a valid dataset file."
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

    @hk.transform_with_state
    def loss_fn(model_config, task_config, inputs, targets, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics),
        )

    def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, jax.random.PRNGKey(0), model_config, task_config, i, t, f
            )
            return loss, (diagnostics, next_state)

        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True
        )(params, state, inputs, targets, forcings)
        return loss, diagnostics, next_state, grads

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

    # loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
    # grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
    run_forward_jitted = drop_state(
        with_params(jax.jit(with_configs(run_forward.apply)))
    )

    print("eval inputs")
    eval_inputs.info()
    print(eval_inputs.time)
    print("eval targets")
    eval_targets.info()
    print(eval_targets.time)

    print("")
    print("eval forcings")
    eval_forcings.info()
    print(eval_forcings.time)

    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )
    print("predictions:")
    predictions.to_netcdf("predictions.nc")
    predictions.info()
    print(predictions.time)


if __name__ == "__main__":
    main()
