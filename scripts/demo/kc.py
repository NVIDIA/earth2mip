import xarray as xr
import numpy as np
from datetime import datetime
from calendar import monthrange
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import random
from datetime import datetime
import mpi4py.MPI as MPI

def load_ens(initial_time, ens):
    path = f"/pscratch/sd/a/amahesh/hens_copy/HENS_summer23_{initial_time:%Y%m%d}T000000/"
    year, month, day = initial_time.year, initial_time.month, initial_time.day
    ds = xr.open_dataset(f"{path}/ensemble_out_{ens:05d}_{year}-{month:02d}-{day:02d}-00-00-00.nc",
                        group='global')[['d2m', 't2m', 'heat_index']]
    
    root = f"/pscratch/sd/a/amahesh/hens_copy/HENS_summer23_{initial_time:%Y%m%d}T000000/ensemble_out_07400_{year}-{month:02d}-{day:02d}-00-00-00.nc"
    ds['time'] = xr.open_dataset(root)['time']
    return ds


if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    print(f"Rank: {rank}, Size: {world_size}")
    kansascity_ens = []
    kansascity_lat, kansascity_lon = 39, 360-94.5

    initial_times = pd.date_range(start='2023-08-09', end='2023-08-23', freq='D')
    for idx, initial_time in enumerate(initial_times):
        if idx % world_size != rank:
            continue
        for ens in range(7424):
            ds = load_ens(initial_time, ens)
            kansascity_ens.append(ds.sel(lat=kansascity_lat,
                                lon=kansascity_lon, time='2023-08-23T18:00:00'))
            if ens % 100 == 0:
                print(ens)
    
        kansascity_ens = xr.concat(kansascity_ens, dim='ensemble')
        kansascity_ens.to_netcdf(f"/pscratch/sd/a/amahesh/hens/kansascity_demo/stlouis_{initial_time:%Y%m%d}.nc")
    
    print("Finished", rank)

