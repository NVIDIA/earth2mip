# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import pathlib
from typing import Optional

import xarray
import xskillscore

import earth2mip.time
from earth2mip import weather_events
from earth2mip.initial_conditions import hdf5

logger = logging.getLogger(__file__)


def save_dataset(out, path):
    out.to_zarr(path + ".zarr", mode="w")


def _open(f, domain, chunks={"time": 1}):
    root = xarray.open_dataset(f, decode_times=True)
    ds = xarray.open_dataset(f, chunks=chunks, group=domain)
    ds.attrs.update(root.attrs)
    return ds.assign_coords(root.coords)


def open_ensemble(path, group):
    path = pathlib.Path(path)
    ensemble_files = sorted(list(path.glob("ensemble_out_*.nc")))
    return xarray.concat([_open(f, group) for f in ensemble_files], dim="ensemble")


def open_verification(time):
    v = hdf5.open_xarray(time)
    v = v.to_dataset("channel")
    v = v.chunk({"time": 1})
    return v


def read_weather_event(dir):
    ncfile = os.path.join(dir, "ensemble_out_0.nc")
    ds = xarray.open_dataset(ncfile)
    weather_event = weather_events.WeatherEvent.parse_raw(ds.weather_event)
    return weather_event


def main(
    input_path: str,
    output_path: Optional[str] = None,
    time_averaging_window: str = "",
    score: bool = True,
    save_ensemble: bool = False,
) -> None:
    if output_path is None:
        output_path = args.input_path

    # use single-threaded scheduler to avoid deadlocks when writing to netCDF.
    # processes doesn't work because locks can't be shared and threaded
    # deadlocks, dask distributed works but isn't any faster, probably because
    # these are I/O bound computations. It is probably better to use zarr as an
    # output.
    pathlib.Path(output_path).mkdir(exist_ok=True)
    weather_event = read_weather_event(input_path)
    for domain in weather_event.domains:
        if domain.type != "Window":
            continue
        ds = open_ensemble(input_path, domain.name)
        ds.attrs["time_averaging_window"] = time_averaging_window
        if time_averaging_window:
            ds = ds.resample(time=time_averaging_window).mean(
                dim="time", keep_attrs=True, skipna=False, keepdims=True
            )

        logger.info("Computing mean")
        ensemble_mean = ds.mean(dim="ensemble", keep_attrs=True)
        save_dataset(ensemble_mean, os.path.join(output_path + "mean"))

        if ds.sizes["ensemble"] > 1:
            logger.info("Computing variance")
            variance = ds.var(dim="ensemble", keep_attrs=True)
            save_dataset(variance, os.path.join(output_path + "variance"))

        if score:
            logger.info("Scoring")
            date_obj = earth2mip.time.convert_to_datetime(ds.time[0])
            v = open_verification(date_obj)
            shared = set(v) & set(ds)
            verification = v[list(shared)]
            ds = ds[list(shared)]
            ds, verification, ensemble_mean = xarray.align(
                ds, verification, ensemble_mean
            )

            if time_averaging_window:
                verification = verification.resample(time=time_averaging_window).mean(
                    dim="time", keep_attrs=True, skipna=False, keepdims=True
                )

            ensemble_mse = (verification - ensemble_mean) ** 2.0
            save_dataset(ensemble_mse, os.path.join(output_path, "ensemble_mse"))

            deterministic_mse = (verification - ds.isel(ensemble=0)) ** 2.0
            deterministic_mse.attrs.update(verification.attrs)
            save_dataset(
                deterministic_mse, os.path.join(output_path, "deterministic_mse")
            )
            crps = xskillscore.crps_ensemble(
                verification,
                ds.chunk(dict(ensemble=-1)),
                issorted=False,
                member_dim="ensemble",
                dim=(),
                keep_attrs=True,
            )
            save_dataset(crps, os.path.join(output_path, "crps"))

        if save_ensemble:
            logger.info("Saving ensemble")
            save_dataset(ds, os.path.join(output_path, "ensemble"))

    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from distributed import Client

    Client()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="full path to the ensemble simulation directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="full path to the ensemble score output directory",
    )
    parser.add_argument(
        "--time_averaging_window",
        type=str,
        help="a string arg for the time averaging as np.datetime64 format, i.e. 2W",
        default="",
    )
    parser.add_argument(
        "--no-score",
        action="store_false",
        dest="score",
        default=True,
        help="Turn off scoring if provided",
    )
    parser.add_argument(
        "--save-ensemble", action="store_true", help="Save out all ensemble members"
    )

    args = parser.parse_args()
    main(
        args.input_path,
        args.output_path,
        args.time_averaging_window,
        args.score,
        args.save_ensemble,
    )
