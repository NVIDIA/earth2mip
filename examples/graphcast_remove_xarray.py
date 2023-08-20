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
from earth2mip.networks.graphcast import channels, inference
from earth2mip.model_registry import Package
import xarray
import datetime
import matplotlib.pyplot as plt
from cartopy import crs

# %%

# on selene
root = "/lustre/fsw/sw_earth2_ml/graphcast/"

# elsewhere
# https://console.cloud.google.com/storage/browser/dm_graphcast/dataset?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false
root = "gs://dm_graphcast"
package = Package(root, seperator="/")
model = inference.Graphcast(package, device="cuda:0")
# %%
import numpy as np
import jax

x = np.ones([1440 * 721, 1, 471])
rng = jax.random.PRNGKey(0)
y = model.run_forward_jitted(rng=rng, x=x)
# %%
dataset_filename = package.get(
    "dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc"
)
with open(dataset_filename, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

# %%
# packing notes
# to graph inputs
# 1. b h c y x -> (y x) b (h c)
# 2. normalize
# 3. x = cat([in, f])
# 4. y = denorm(f(x)) + x

# %%
model.task_config
# %%
from earth2mip.initial_conditions import cds
from graphcast import data_utils
import numpy as np

task_config = model.task_config
x_codes = channels.get_codes(
    task_config.input_variables, levels=task_config.pressure_levels, time_levels=[0, 1]
)
f_codes = channels.get_codes(
    task_config.forcing_variables, levels=task_config.pressure_levels, time_levels=[2]
)
t_codes = channels.get_codes(
    task_config.target_variables, levels=task_config.pressure_levels, time_levels=[2]
)


arrays = []
time = datetime.datetime(2018, 1, 1)
seconds = time.timestamp()
s = model.history_time_step.total_seconds()
seconds_since_epoch = np.array([seconds + (i - 1) * s for i in range(3)])
seconds_since_epoch = seconds_since_epoch.reshape([3])
day_progress = data_utils.get_day_progress(seconds_since_epoch, model.grid.lon).reshape(
    [1, 3, 1, 1, 1440]
)
year_progress = data_utils.get_year_progress(seconds_since_epoch).reshape(
    [1, 3, 1, 1, 1]
)


def add_toa(d):
    tisr = [
        channels.toa_incident_solar_radiation(
            time + (i - 1) * model.history_time_step,
            model.grid.lat[:, None],
            model.grid.lon[None, :],
        )
        for i in range(3)
    ]

    # h y x
    tisr = np.stack(tisr)
    d["toa_incident_solar_radiation"] = tisr[None, :, None]


channel_shape = [1, 3, 1, *model.grid.shape]
day_progress = np.broadcast_to(day_progress, channel_shape)
year_progress = np.broadcast_to(day_progress, channel_shape)

channel_shape = [1, 1, *model.grid.shape]
lsm = np.broadcast_to(example_batch.land_sea_mask.values[None, None], channel_shape)
zs = np.broadcast_to(
    example_batch.geopotential_at_surface.values[None, None], channel_shape
)


forcings = {
    "day_progress_sin": np.sin(day_progress),
    "day_progress_cos": np.cos(day_progress),
    "year_progress_sin": np.sin(year_progress),
    "year_progress_cos": np.cos(year_progress),
}
add_toa(forcings)

for code in x_codes:
    match code:
        case t, cds.PressureLevelCode(id, level):
            arr = (
                example_batch[channels.CODE_TO_GRAPHCAST_NAME[id]]
                .sel(level=level)
                .values[:, t, None]
            )
        case t, cds.SingleLevelCode(id):
            arr = example_batch[channels.CODE_TO_GRAPHCAST_NAME[id]].values[:, t, None]
        case "land_sea_mask":
            arr = lsm
        case "geopotential_at_surface":
            arr = zs
        case t, str(s):
            arr = forcings[s][:, t]
    arrays.append(arr)

for code in f_codes:
    match t, code:
        case cds.SingleLevelCode(id):
            arr = example_batch[channels.CODE_TO_GRAPHCAST_NAME[id]].values[:, t, None]
        case str(s):
            arr = forcings[s][:, t]
    arrays.append(arr)

array = np.concatenate(arrays, axis=1)
import einops

array = einops.rearrange(array, "b c y x -> (y x) b c")


def get_data_for_code_scalar(code, scalar):
    match code:
        case _, cds.PressureLevelCode(id, level):
            arr = scalar[channels.CODE_TO_GRAPHCAST_NAME[id]].sel(level=level).values
        case _, cds.SingleLevelCode(id):
            arr = scalar[channels.CODE_TO_GRAPHCAST_NAME[id]].values
        case "land_sea_mask":
            arr = scalar[code].values
        case "geopotential_at_surface":
            arr = scalar[code].values
        case _, str(s):
            arr = scalar[s].values
    return arr


mean = np.array(
    [get_data_for_code_scalar(code, model.mean_by_level) for code in x_codes + f_codes]
)
scale = np.array(
    [
        get_data_for_code_scalar(code, model.stddev_by_level)
        for code in x_codes + f_codes
    ]
)
diff_scale = np.array(
    [get_data_for_code_scalar(code, model.diffs_stddev_by_level) for code in t_codes]
)


in_codes = x_codes + f_codes

prog_level_0 = [in_codes.index((0, c)) for _, c in t_codes]
prog_level_1 = [in_codes.index((1, c)) for _, c in t_codes]
forcing_level_0 = [in_codes.index((0, v)) for v in forcings]
forcing_level_1 = [in_codes.index((1, v)) for v in forcings]
forcing_level_2 = [in_codes.index((2, v)) for v in forcings]

# %%
x = (array - mean) / scale
d = model.run_forward_jitted(rng=rng, x=x)
x_next = x[:, :, prog_level_1] * mean[prog_level_1] + d * diff_scale

array[:, :, prog_level_0] = array[:, :, prog_level_1]
array[:, :, prog_level_1] = x_next


# %%
names = [str(c) for _, c in t_codes]
i = names.index("q925")
import jax.dlpack
import torch

p = jax.dlpack.to_dlpack(x_next)
pt = torch.from_dlpack(p)
pt = pt[:, :, i].reshape([*model.grid.shape, -1])
plt.pcolormesh(pt[:, :, 0].cpu().numpy())

# %%
