# flake8: noqa
# TODO delete
# %%
import sys

sys.path.insert(0, "../tests")
from test_graphcast import test_tisr_matches_cds
from earth2mip.networks import graphcast
from earth2mip.networks.graphcast import irradiance

from graphcast.graphcast import TASK, TASK_13_PRECIP_OUT, TASK_13
from earth2mip.initial_conditions import cds
import torch
import datetime
import numpy as np
import xarray

import pytest

from earth2mip.model_registry import Package


cds_data, compute = test_tisr_matches_cds()
# %%

# %% CDS
import cdsapi


grid = (0.25, 0.25)
area = (90, -180, -90, 180)
format = "netcdf"


client = cdsapi.Client()

if not os.path.isfile("out.nc"):
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [212],
            "year": 2018,
            "month": 1,
            "day": 1,
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "grid": grid,
            "format": format,
        },
        "out.nc",
    )

# %% The time mean is zonally uniform
tisr = xarray.open_dataset("out.nc")
tisr

s = tisr.tisr.sum("time") / 24 / 3600
s.plot()
s.std("longitude") / s.mean("longitude")  # deviation is 1e-4


# %%
def global_average(x):
    w = np.cos(np.deg2rad(x.latitude))
    return x.weighted(w).mean(["latitude", "longitude"])


s = tisr / 3600
total_tisr = global_average(s)
print(total_tisr)

# %%


def irradiance_simplified(t, S0=1361, e=0.0167, perihelion_longitude=281.183):
    """The flux of solar energy in W/m2 towards Earth

    The default orbital parameters are set to 2000 values.
    Over the period of 1900-2100 this will result in an error of at most 0.02%,
    so can be neglected for many applications.

    Args:
        t: linux timestamp
        S0: the solar constant in W/m2. This is the mean irradiance received by
            earth over a year.
        e: the eccentricity of earths elliptical orbit
        perihelion_longitude: spatial angle from moving vernal equinox to perihelion with Sun as angle vertex.
            Perihelion is moment when earth is closest to sun. vernal equinox is
            the longitude when the Earth crosses the equator from South to North.

    """

    year = 365.25
    seconds_in_day = 86400
    seconds_in_year = 86400 * year

    day = (t - datetime.datetime(2018, 1, 1).timestamp()) / seconds_in_year
    day = day % 1

    return S0 * (1 + 0.034 * np.cos(day * 2 * np.pi))


time = datetime.datetime(2018, 1, 1, 23)

I = irradiance(time.timestamp())
I2100 = irradiance(time.timestamp(), perihelion_longitude=281.183)
I1900 = irradiance(time.timestamp(), perihelion_longitude=284.609, e=0.0166)
print(I / 4, (I2100 - I1900) / I2100 * 100)


# %% CDS
import cdsapi

file = "monthly.nc"
if not os.path.isfile(file):
    grid = (0.25, 0.25)
    area = (90, -180, -90, 180)
    format = "netcdf"

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [212],
            "year": [2018, 2019],
            "month": list(range(1, 13)),
            "day": 1,
            "time": "00:00",
            "area": area,
            "grid": grid,
            "format": format,
        },
        file,
    )

monthly = xarray.open_dataset(file).tisr

# %%


def to_timestamp(numpy_datetime: np.ndarray):
    return (numpy_datetime - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(
        1, "s"
    )


t = to_timestamp(monthly.time)
irr = irradiance(t)
irr_simple = irradiance(t, perihelion_longitude=284)
era5 = global_average(monthly / 3600)

# irr.plot(label='computed')
irr_simple.plot(label="computed")
(era5 * 4).plot(label="from ERA5")


# %%
from modulus.utils.sfno.zenith_angle import cos_zenith_angle
import matplotlib.pyplot as plt

time = datetime.datetime(2018, 1, 1)
mu = cos_zenith_angle(time, monthly.longitude, monthly.latitude)
tisr_compute = np.maximum(mu, 0) * irradiance(time.timestamp())
e5 = tisr.sel(time=time).tisr / 3600


e5.sel(latitude=0).plot(label="era5")
tisr_compute.sel(latitude=0).plot(label="tisr")
plt.legend()
