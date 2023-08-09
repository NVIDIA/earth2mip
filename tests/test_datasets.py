import xarray as xr
from earth2mip.datasets import zarr_directory, hindcast
import numpy as np
import os

import pytest


def test_zarr_directory():

    # Generate data filled with ones
    data_a = np.ones((3, 4))
    data_b = np.arange(4)

    # Create xarray dataset
    ds = xr.Dataset(
        {
            "a": (["x", "y"], data_a),
            "b": (["y"], data_b),
        },
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    store = {}
    ds.to_zarr(store)
    nested_store = {}
    directories = [str(i) for i in range(10)]
    for i in directories:
        for key in store:
            # TODO remove the hardcode
            nested_store[f"{i}/mean.zarr/" + key] = store[key]

    obj = zarr_directory.NestedDirectoryStore(
        nested_store,
        directories=directories,
        concat_dim="dim",
        group="mean.zarr",
        static_coords=("x", "y"),
        dim_rename=None,
    )
    loaded = xr.open_zarr(obj)

    for val in directories:
        iobj = loaded.sel(dim=val).load()
        for v in iobj:
            xr.testing.assert_equal(iobj.variables[v], ds.variables[v])


def test_open_forecast():
    root = "/lustre/fsw/sw_climate_fno/nbrenowitz/scoring_tools/sfno_coszen/deterministic-medium"  # noqa
    if not os.path.isdir(root):
        pytest.skip()

    ds = hindcast.open_forecast(root, group="mean.zarr")
    print(ds)
    ds.z500
