"""
pip install -r earth2mip/networks/graphcast/requirements.txt


Run like::

    python3 -m examples.graphcast

to avoid conflicting with the google graphcast library

TODO
- currently this script does not work for time-steps > 1, something is broken in
  the graphcast wrapper.
"""
# %%
import sys

sys.path.insert(0, "..")
import datetime

from earth2mip.time_loop import TimeLoop
import torch
import jax
import jax.dlpack
import einops
import matplotlib.pyplot as plt
import numpy as np
import xarray
from graphcast import data_utils

from earth2mip.initial_conditions import cds
from earth2mip.model_registry import Package
from earth2mip.networks.graphcast import channels, inference


def torch_to_jax(x):
    return jax.dlpack.from_dlpack(torch.to_dlpack(x))


class GrapchastTimeLoop(TimeLoop):
    """
    # packing notes
    # to graph inputs
    # 1. b h c y x -> (y x) b (h c)
    # 2. normalize
    # 3. x = cat([in, f])
    # 4. y = denorm(f(x)) + x

    """

    def __init__(self, package):
        self.model = inference.Graphcast(package, device="cuda:0")
        dataset_filename = package.get(
            "dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc"
        )
        with open(dataset_filename, "rb") as f:
            example_batch = xarray.load_dataset(f).compute()

        in_codes, t_codes = channels.get_codes_from_task_config(self.model.task_config)

        self._static_variables = {
            key: example_batch[key].values
            for key in ["land_sea_mask", "geopotential_at_surface"]
        }

        self.mean = np.array(
            [
                channels.get_data_for_code_scalar(code, self.model.mean_by_level)
                for code in in_codes
            ]
        )
        self.scale = np.array(
            [
                channels.get_data_for_code_scalar(code, self.model.stddev_by_level)
                for code in in_codes
            ]
        )
        self.diff_scale = np.array(
            [
                channels.get_data_for_code_scalar(
                    code, self.model.diffs_stddev_by_level
                )
                for code in t_codes
            ]
        )

        self.in_codes = in_codes
        self.prog_levels = [
            [in_codes.index((t, c)) for _, c in t_codes] for t in range(2)
        ]
        self.out_channel_names = [str(c) for _, c in t_codes]

    @property
    def grid(self):
        return self.model.grid

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
        model = self.model

        lat, lon = np.meshgrid(model.grid.lat, model.grid.lon, indexing="ij")

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
        array = einops.rearrange(
            array, "(y x) b c -> b c y x", y=self.model.grid.shape[0]
        )
        p = jax.dlpack.to_dlpack(array)
        pt = torch.from_dlpack(p)
        return pt

    def _input_codes(self):
        return list(get_codes(self.model.task_config))

    @property
    def in_channel_names(self):
        return [str(c) for c in get_codes(task_config)]

    def __call__(self, time, x, restart=None):
        assert not restart, "not implemented"
        ngrid = np.prod(self.model.grid.shape)
        array = torch.empty([ngrid, 1, len(self.in_codes)], device=x.device)

        # set input data
        x_codes = self._input_codes()
        for t in range(2):
            index_in_input = [self.in_codes.index((t, c)) for c in x_codes]
            array[:, :, index_in_input] = einops.rearrange(
                x[:, t], "b c y x -> (y x) b c"
            )

        self.set_static_variables(array)

        rng = jax.random.PRNGKey(0)
        s = torch_to_jax(array)

        while True:
            # TODO will need to change update rule for diagnostics outputs
            yield time, self._to_latlon(self.get_prognostic(s, 1)), None

            s = self.set_forcings(s, time - 1 * self.model.history_time_step, 0)
            s = self.set_forcings(s, time, 1)
            s = self.set_forcings(s, time + self.model.history_time_step, 2)

            x = (s - self.mean) / self.scale
            d = self.model.run_forward_jitted(rng=rng, x=x)
            x_next = self.get_prognostic(s, 1) + d * self.diff_scale

            # update array
            s = self.set_prognostic(s, 0, self.get_prognostic(s, 1))
            s = self.set_prognostic(s, 1, x_next)
            time = time + self.model.time_step


def get_input_from_xarray(task_config, example_batch):
    arrays = []
    levels = list(task_config.pressure_levels)
    for v in task_config.target_variables:
        if channels.is_3d(v):
            # b, h, p, y, x
            arr = example_batch[v].sel(level=levels).isel(time=slice(0, 2)).values
            assert arr.ndim == 5
            arrays.append(arr)
        else:
            arr = example_batch[v].isel(time=slice(0, 2)).values
            assert arr.ndim == 4
            arr = np.expand_dims(arr, 2)
            arrays.append(arr)

    return np.concatenate(arrays, axis=2)


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


# %%

# on selene
root = "/lustre/fsw/sw_earth2_ml/graphcast/"

# elsewhere
# https://console.cloud.google.com/storage/browser/dm_graphcast/dataset?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false
root = "gs://dm_graphcast"
package = Package(root, seperator="/")
time_loop = GrapchastTimeLoop(package)

dataset_filename = package.get(
    "dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc"
)
with open(dataset_filename, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

# %%


task_config = time_loop.model.task_config
target_codes = channels.get_codes(
    task_config.target_variables, task_config.pressure_levels, [0]
)
array = get_input_from_xarray(task_config, example_batch)
pt = torch.from_numpy(array).cuda()

time = datetime.datetime(2018, 1, 1)
for i, (time, x, _) in enumerate(time_loop(time, pt)):
    print(i)
    if i > 10:
        break


# %%
i = time_loop.out_channel_names.index("z500")
plt.pcolormesh(x[0, i].cpu().numpy())

# %%
