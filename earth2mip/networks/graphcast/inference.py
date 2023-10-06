# %%
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
        self.prog_levels = [
            [in_codes.index((t, c)) for _, c in t_codes] for t in range(2)
        ]
        self.in_channel_names = self.out_channel_names = [str(c) for _, c in t_codes]

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
        d = self.forward(rng=rng, x=x)
        x_next = self.get_prognostic(s, 1) + d * self.diff_scale

        # update array
        s = self.set_prognostic(s, 0, self.get_prognostic(s, 1))
        s = self.set_prognostic(s, 1, x_next)
        return s

    def __call__(self, time, x, restart=None):
        assert not restart, "not implemented"
        ngrid = len(self.lon) * len(self.lat)
        array = torch.empty([ngrid, 1, len(self.in_codes)], device=x.device)

        # set input data
        x_codes = self._input_codes()
        for t in range(2):
            index_in_input = [self.in_codes.index((t, c)) for c in x_codes]
            array[:, :, index_in_input] = einops.rearrange(
                torch.flip(x[:, t], [-2]), "b c y x -> (y x) b c"
            )

        self.set_static_variables(array)

        rng = jax.random.PRNGKey(0)
        s = torch_to_jax(array)

        while True:
            # TODO will need to change update rule for diagnostics outputs
            yield time, self._to_latlon(self.get_prognostic(s, 1)), None
            s = self.step(rng, time, s)
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


def load_time_loop(package, pretrained=True, device="cuda:0"):
    def join(*args):
        return package.get(os.path.join(*args))

    model_name = "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"

    checkpoint_path = join("params", model_name)
    # load checkpoint:
    with open(checkpoint_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
        task_config = ckpt.task_config
        run_forward = load_run_forward_from_checkpoint(
            ckpt, grid=GraphcastTimeLoop.grid
        )

    dataset_filename = package.get(
        "dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc"
    )
    with open(dataset_filename, "rb") as f:
        example_batch = xarray.load_dataset(f).compute()

    in_codes, t_codes = channels.get_codes_from_task_config(task_config)

    # declare static variables
    static_variables = {
        key: example_batch[key].values
        for key in ["land_sea_mask", "geopotential_at_surface"]
    }

    # load stats
    with open(join("stats/diffs_stddev_by_level.nc"), "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with open(join("stats/mean_by_level.nc"), "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open(join("stats/stddev_by_level.nc"), "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()

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
