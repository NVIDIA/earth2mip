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


time = datetime.datetime(2018, 1, 1)
channel_shape = [1, 1, *model.grid.shape]
import einops


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

# %%
ngrid = np.prod(model.grid.shape)
array = np.empty([ngrid, 1, len(in_codes)], dtype=np.float32)

prog_level_0 = [in_codes.index((0, c)) for _, c in t_codes]
prog_level_1 = [in_codes.index((1, c)) for _, c in t_codes]


for k, code in enumerate(x_codes):
    match code:
        case t, cds.PressureLevelCode(id, level):
            arr = (
                example_batch[channels.CODE_TO_GRAPHCAST_NAME[id]]
                .sel(level=level)
                .values[:, t, None]
            )
        case t, cds.SingleLevelCode(id):
            arr = example_batch[channels.CODE_TO_GRAPHCAST_NAME[id]].values[:, t, None]
    array[:, 0, k] = einops.rearrange(arr, "b h y x ->  h (b y x)")


def set_static(field, example_batch):
    assert example_batch[field].dims == ("lat", "lon")
    arr = example_batch[field].values
    k = in_codes.index(field)
    array[:, 0, k] = einops.rearrange(arr, "y x ->  (y x)")


def set_forcing(v, t, data):
    # (y x) b c
    i = in_codes.index((t, v))
    array[:, :, i] = data


def set_forcings(time: datetime.datetime, t: int):
    seconds = time.timestamp()

    lat, lon = np.meshgrid(model.grid.lat, model.grid.lon, indexing="ij")

    lat = lat.reshape([-1, 1])
    lon = lon.reshape([-1, 1])

    day_progress = data_utils.get_day_progress(seconds, lon)
    year_progress = data_utils.get_year_progress(seconds)
    set_forcing("day_progress_sin", t, np.sin(day_progress))
    set_forcing("day_progress_cos", t, np.cos(day_progress))
    set_forcing("year_progress_sin", t, np.sin(year_progress))
    set_forcing("year_progress_cos", t, np.cos(year_progress))

    tisr = channels.toa_incident_solar_radiation(time, lat, lon)
    set_forcing("toa_incident_solar_radiation", t, tisr)


def set_prognostic(t: int, data):
    index = [prog_level_0, prog_level_1][t]
    array[:, :, index] = data


def get_prognostic(t: int):
    index = [prog_level_0, prog_level_1][t]
    return array[:, :, index]


set_static("land_sea_mask", example_batch)
set_static("geopotential_at_surface", example_batch)
rng = jax.random.PRNGKey(0)

for i in range(5):
    print(i)
    set_forcings(time - 1 * model.history_time_step, 0)
    set_forcings(time, 1)
    set_forcings(time + model.history_time_step, 2)

    x = (array - mean) / scale
    d = model.run_forward_jitted(rng=rng, x=x)
    x_next = array[:, :, prog_level_1] + d * diff_scale

    # update array
    set_prognostic(0, get_prognostic(1))
    set_prognostic(1, x_next)
    time = time + model.time_step

# %%
set_prognostic(0, get_prognostic(1))
get_prognostic(1)


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
