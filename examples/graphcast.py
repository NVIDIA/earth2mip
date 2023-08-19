"""
pip install -r earth2mip/networks/graphcast/requiremnets.txt


Run like::

    python3 -m examples.graphcast

to avoid conflicting with the google graphcast library

TODO
- currently this script does not work for time-steps > 1, something is broken in
  the graphcast wrapper.
"""
# %%
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
# about 2 GB of data, so expect this to take some time
# it will be locally cached, so you can evaluate it
dataset_filename = package.get(
    "dataset/source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc"
)
with open(dataset_filename, "rb") as f:
    example_batch = xarray.load_dataset(f).compute()

packed = channels.pack(example_batch, model.codes)
# only use first two input levels
packed = packed[:, : model.n_history_levels]
# graphcast uses -90 to 90 for lat, need to rev to put on our grid
packed = packed[:, :, :, ::-1]
time = datetime.datetime(2022, 1, 1)

# %%
i = model.out_channel_names.index("t2m")
fig = plt.figure()
for k, (t, y) in enumerate(model(time, packed)):
    if k > 5:
        break
    print(t)
    plt.clf()
    ax = fig.add_subplot(projection=crs.PlateCarree())
    ax.pcolormesh(model.grid.lon, model.grid.lat, y[0, i], transform=crs.PlateCarree())
    ax.coastlines(color="w")
    plt.savefig(f"{k:04d}.png")
