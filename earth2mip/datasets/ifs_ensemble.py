import pandas as pd
import numpy as np
import os
import datetime
import xarray
import s3fs
import fsspec


gravity = 9.807


def _get_fs(path):
    if path.startswith("s3://"):
        return s3fs.S3FileSystem(client_kwargs=dict(endpoint_url="https://pbss.s8k.io"))
    else:
        return fsspec.filesystem("file")


def _open(f, initial):
    ds = xarray.open_dataset(f, chunks=None)
    ds = ds.rename(
        phony_dim_0="initial_time",
        phony_dim_1="time",
        phony_dim_2="ensemble",
        phony_dim_3="lat",
        phony_dim_4="lon",
    )
    ds = ds.rename(forecast_lead_times="time")
    time = ds.era5_steps * datetime.timedelta(hours=6)
    ds = ds.assign_coords(initial_time=pd.Timestamp(initial) + time[:, 0])
    ds["time"] = ds.time * datetime.timedelta(hours=1)
    ds = ds.drop("era5_steps")

    nlat = ds.sizes["lat"]
    nlon = ds.sizes["lon"]
    lat = 90 - np.arange(nlat) * 0.25
    lon = np.arange(nlon) * 0.25

    return ds.assign_coords(lat=lat, lon=lon)


def _open_tigge(variable_path):
    # month 12 is corrupted/missing
    filenames = [f"{variable_path}/2018_{m}.h5" for m in range(1, 12)]
    dss = [
        _open(f, datetime.datetime(2018, 1, 1)).chunk(ensemble=-1, initial_time=1)
        for f in filenames
    ]
    ds = xarray.concat(dss, dim="initial_time")
    return ds


def open_tigge(root):
    # month 12 is corrupted/missing
    ds = xarray.Dataset()
    # ifs is in m rather than geopontential
    ds["z500"] = _open_tigge(os.path.join(root, "z500"))["ifs_ensemble"] * gravity
    ds["u10m"] = _open_tigge(os.path.join(root, "u10"))["ifs_ensemble"]
    ds["valid_time"] = ds.time + ds.initial_time
    return ds
