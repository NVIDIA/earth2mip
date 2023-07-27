from earth2mip import config
import xarray
import datetime
from earth2mip import schema
import joblib
import numpy as np
from earth2mip.initial_conditions.era5 import open_era5_xarray
from earth2mip.initial_conditions import ifs
from earth2mip.initial_conditions import cds
from earth2mip.initial_conditions import gfs
from earth2mip.initial_conditions import hrmip

# hack...shouldn't be imported from here
from earth2mip.datasets.era5 import METADATA
import json

__all__ = ["open_era5_xarray", "get"]


def get(
    n_history: int,
    time: datetime.datetime,
    channel_set: schema.ChannelSet,
    source: schema.InitialConditionSource = schema.InitialConditionSource.era5,
) -> xarray.DataArray:
    if source == schema.InitialConditionSource.era5:
        ds = open_era5_xarray(time, channel_set)
        subset = ds.sel(time=slice(None, time))
        subset = subset[-n_history - 1 :]
        num_time = subset.sizes["time"]
        if num_time != n_history + 1:
            a = ds.time.min().values
            b = ds.time.max().values
            raise ValueError(
                f"{num_time} found. Expected: {n_history + 1} ."
                f"Time requested: {time}. Time range in data: {a} -- {b}."
            )
        return subset.load()
    elif source == schema.InitialConditionSource.hrmip:
        ds = hrmip.get(time, channel_set)
        return ds
    elif source == schema.InitialConditionSource.ifs:
        if n_history > 0:
            raise NotImplementedError("IFS initializations only work with n_history=0.")
        ds = ifs.get(time, channel_set)
        ds = ds.expand_dims("time", axis=0)
        # move to earth2mip.channels

        # TODO refactor interpolation to another place
        metadata = json.loads(METADATA.read_text())
        lat = np.array(metadata["coords"]["lat"])
        lon = np.array(metadata["coords"]["lon"])
        ds = ds.roll(lon=len(ds.lon) // 2, roll_coords=True)
        ds["lon"] = ds.lon.where(ds.lon >= 0, ds.lon + 360)
        assert min(ds.lon) >= 0, min(ds.lon)
        return ds.interp(lat=lat, lon=lon, kwargs={"fill_value": "extrapolate"})
    elif source == schema.InitialConditionSource.cds:
        if n_history > 0:
            raise NotImplementedError("CDS initializations only work with n_history=0.")
        ds = cds.get(time, channel_set)
        return ds
    elif source == schema.InitialConditionSource.gfs:
        if n_history > 0:
            raise NotImplementedError("GFS initializations only work with n_history=0.")
        return gfs.get(time, channel_set)
    else:
        raise NotImplementedError(source)


if config.LOCAL_CACHE:
    memory = joblib.Memory(config.LOCAL_CACHE)
    get = memory.cache(get)


def ic(
    time: datetime,
    grid,
    n_history: int,
    channel_set: schema.ChannelSet,
    source: schema.InitialConditionSource,
):
    ds = get(n_history, time, channel_set, source)
    # TODO collect grid logic in one place
    if grid == schema.Grid.grid_720x1440:
        return ds.isel(lat=slice(0, -1))
    elif grid == schema.Grid.grid_721x1440:
        return ds
    else:
        raise NotImplementedError(f"Grid {grid} not supported")
