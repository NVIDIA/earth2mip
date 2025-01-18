print("opening python script")
import xarray as xr
from datetime import datetime, timedelta
import os
import numpy as np
import pathlib
import pandas as pd
from dask.distributed import Client
import argparse


def find_nearest_reforecast_dates(month, day):
    """
    Let's say 2003 to 2022 are the reforecast years.
    2023 is the validation year.

    Let's say the REFORECAST dates are on the nearest Monday
    and Thursday to a given date (this is how ECMWF does it.)

    This method takes in a month and day of 2023. This method
    returns the 9 nearest REFORECAST dates
    in each of the reforecast years.
    """
    # Use year 2023 since that is the validation year
    given_date = datetime(2023, month, day)

    start_date = datetime(2003, 1, 1)
    end_date = datetime(2022, 12, 31)
    delta = timedelta(days=1)

    reforecast_dates = []

    while start_date <= end_date:
        if start_date.weekday() in [0, 3]:  # Monday is 0 and Thursday is 3
            reforecast_dates.append(start_date)
        start_date += delta

    nearest_dates = []

    for year in range(2003, 2023):
        # Find the 9 nearest dates in year to the given_date
        nearest_dates_year = sorted(
            reforecast_dates,
            key=lambda d: abs((given_date.replace(year=year) - d).days),
        )[:9]
        nearest_dates.extend(nearest_dates_year)

    return nearest_dates

def open_files(month, day, root):
    reforecast_dates = find_nearest_reforecast_dates(month, day)

    ensembles = []
    files_used = []
    for date in reforecast_dates:
        file_str = f"{root}/{date:%Y-%m-%d}T00:00:00/ensemble.zarr"
        files_used.append(file_str)
        curr_zarr = xr.open_zarr(file_str, decode_times=False)
        ensembles.append(curr_zarr)
    
    return xr.concat(ensembles, dim="ensemble"), files_used


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script calculates the percentiles of the mclimate for a given day"
    )
    parser.add_argument(
        "--task_id", type=int, help="the task id, which corresponds to the date"
    )
    parser.add_argument(
        "--output_path", type=str, help="the location at which to save the percentiles"
    )
    parser.add_argument(
        "--mclimate_path", type=str, help="the location of the mclimate path"
    )
    args = parser.parse_args()

    print("starting percentiles calculation")
    scheduler_file = os.path.join(
        args.output_path, "scheduler_file_{}.json".format(args.task_id)
    )
    client = Client(scheduler_file=scheduler_file)

    all_dates = pd.date_range("2023-06-01", "2023-08-31", freq="D")
    date = all_dates[args.task_id]

    ds, files_used = open_files(date.month, date.day, args.mclimate_path)
    lead_time = pd.date_range(
        f"2023-{date.month:02d}-{date.day:02d}", freq="6H", periods=ds["time"].shape[0]
    )
    ds = ds.assign_coords(time=lead_time)
    for mode in ["mean", "max"]:
        if mode == "mean":
            resampled_ds = ds.isel(time=slice(0, 60)).resample(time="D").mean()
        else:
            resampled_ds = ds.isel(time=slice(0, 60)).resample(time="D").max()
        
        percentiles = np.arange(0.01, 1.0, 0.02)
        percentiles = np.concatenate([[0.0001, 0.001], percentiles, [1-0.001, 1-0.0001]])

        ds_percentiles = resampled_ds.chunk({"ensemble": -1}).quantile(percentiles, dim="ensemble")
        ds_percentiles.attrs["files used"] = str(files_used)

        ds_percentiles.to_zarr(
            f"{args.output_path}/MClimate_{mode}_Percentiles.2023-{date.month:02d}-{date.day:02d}.zarr",
            mode="w",
        )
        print("completed percentiles calculation")
