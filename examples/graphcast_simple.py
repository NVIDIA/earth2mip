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

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray

from earth2mip.model_registry import Package
from earth2mip.networks.graphcast import channels, inference


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


# %%

# on selene
root = "/lustre/fsw/sw_earth2_ml/graphcast/"

# elsewhere
# https://console.cloud.google.com/storage/browser/dm_graphcast/dataset?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false
root = "gs://dm_graphcast"
package = Package(root, seperator="/")
time_loop = inference.load_time_loop(package)

dataset_filename = package.get(
    "dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc"
)
with open(dataset_filename, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

# %%


task_config = time_loop.task_config
target_codes = channels.get_codes(
    task_config.target_variables, task_config.pressure_levels, [0]
)
array = get_input_from_xarray(task_config, example_batch)
pt = torch.from_numpy(array).cuda()

time = datetime.datetime(2018, 1, 1)
i = time_loop.out_channel_names.index("q925")

for k, (time, x, _) in enumerate(time_loop(time, pt)):
    print(k)
    plt.pcolormesh(x[0, i].cpu().numpy())
    plt.savefig(f"{k:03d}.png")
    if k == 10:
        break
