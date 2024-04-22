from typing import Optional
import numpy as np
import os
import argparse
import xarray
import pathlib
import earth2mip.time
from earth2mip.initial_conditions import hdf5
from earth2mip import initial_conditions, weather_events, schema
import xskillscore
import xhistogram.xarray
from earth2mip.extreme_scoring_utils import (
    set_threshold,
    threshold_at_extremes,
    compute_log_score,
    compute_efi,
    compute_edi,
    compute_brier_score_decomposition,
)
import logging

logger = logging.getLogger(__file__)


def save_dataset(out, path):
    out.to_zarr(path + ".zarr", mode="w")


def _open(f, domain, chunks={"time": 4}):
    root = xarray.open_dataset(f, decode_times=True)
    ds = xarray.open_dataset(f, chunks=chunks, group=domain)
    ds.attrs.update(root.attrs)
    return ds.assign_coords(root.coords)


def open_ensemble(path, group):
    path = pathlib.Path(path)
    ds = xarray.open_zarr("{}/ensemble.zarr".format(path))
    return ds

def open_ensemble_OLD(path, group):
    path = pathlib.Path(path)
    ensemble_files = sorted(list(path.glob("ensemble_out_*.nc")))
    ds = xarray.concat([_open(f, group) for f in ensemble_files], dim="ensemble")
    ens_size = ds["ensemble"].shape[0]
    return ds.chunk({"ensemble": min(ens_size, 52)})


def open_ifs(time):
    ds = xarray.open_zarr('gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr')
    subset = ds[['2m_temperature']].sel(time=time)
    subset = subset.rename({'latitude' : 'lat',
                    'longitude' : 'lon',
                    'time' : 'initial_time',
                    'number' : 'ensemble'})
    subset['prediction_timedelta'] = subset['prediction_timedelta'] + subset['initial_time']
    subset = subset.rename({'prediction_timedelta' : 'time',
                            '2m_temperature' : 't2m'})
    return subset.chunk({"time": 1,
                         "lat": 721,
                         "lon": 1440})


def open_operational_analysis():
    ds = xarray.open_zarr('gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr')
    subset = ds[['2m_temperature']]
    subset = subset.rename({'latitude' : 'lat',
                    'longitude' : 'lon'})
    subset = subset.rename({'2m_temperature' : 't2m'})
    return subset.chunk({"time": 1,
                         "lat": 721,
                         "lon": 1440})

def open_verification(time):
    v = hdf5.open_xarray(time)
    v = v.to_dataset("channel")
    v = v.chunk({"time": 1})
    return v

def read_weather_event(dir):
    ncfile = os.path.join(dir, "ensemble_out_00000_*.nc")
    ds = xarray.open_mfdataset(ncfile)
    weather_event = weather_events.WeatherEvent.parse_raw(ds.weather_event)
    return weather_event

def main(
    input_path: str,
    output_path: Optional[str] = None,
    time_averaging_window: str = "",
    score: bool = True,
    save_ensemble: bool = False,
    ifs: bool = None,
    extreme_scoring: str = None,
    efi_path: str = None,
    precip : bool = False
) -> None:

    if output_path is None:
        output_path = args.input_path

    # use single-threaded scheduler to avoid deadlocks when writing to netCDF.
    # processes doesn't work because locks can't be shared and threaded
    # deadlocks, dask distributed works but isn't any faster, probably because
    # these are I/O bound computations. It is probably better to use zarr as an
    # output.
    pathlib.Path(output_path).mkdir(exist_ok=True)
    if ifs:
        # TODO: add support for specifying a domain in IFS
        domains = ["global"]
    else:
        #weather_event = read_weather_event(input_path)
        #domains = weather_event.domains
        domains = ["global"]


    for domain in domains:
        #if not ifs and domain.type != "Window":
        #    continue

        if ifs:
            ds = open_ifs(input_path)
        else:
            ds = open_ensemble(input_path, "") #domain.name)
        
        ds.attrs["time_averaging_window"] = time_averaging_window
        if time_averaging_window:
            ds = ds.resample(time=time_averaging_window).mean(
                dim="time", keep_attrs=True, skipna=False, #keepdims=True
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
            if ifs:
                v = open_operational_analysis()
            else:
                v = open_verification(date_obj)
            shared = set(v) & set(ds)
            verification = v[list(shared)]
            ds = ds[list(shared)]
            ds, verification, ensemble_mean = xarray.align(
                ds, verification, ensemble_mean
            )

            if time_averaging_window:
                verification = verification.resample(time=time_averaging_window).mean(
                    dim="time", keep_attrs=True, skipna=False, #keepdims=True
                )

            ensemble_mse = (verification - ensemble_mean) ** 2.0
            ensemble_mse.attrs.update(variance.attrs)
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
            crps.attrs = variance.attrs
            save_dataset(crps, os.path.join(output_path, "crps"))

            if extreme_scoring is None:
                continue
            if precip:
                variables = ['tp']
            else:
                variables = ["t2m"] # "tcwv", "wind_speed10m"]
            ds = ds[variables]
            verification = verification[variables]

            for percentile in [0.99]: #, 0.99]:
                output_path_percentile = os.path.join(
                    output_path, f"percentile_{percentile}/"
                )
                threshold = set_threshold(
                    list(ds.keys()), ds['time'], percentile, extreme_scoring,
                    ds, window=time_averaging_window
                )
                thresholded_ds = threshold_at_extremes(ds, threshold)
                thresholded_ensemble_mean = thresholded_ds.mean("ensemble")
                thresholded_verification = xarray.where(verification > threshold, 1, 0)
                save_dataset(
                    thresholded_verification,
                    os.path.join(output_path_percentile, "thresholded_verification"),
                )
                save_dataset(
                    thresholded_ensemble_mean,
                    os.path.join(output_path_percentile, "thresholded_ensemble_mean"),
                )
                thresholded_verification = xarray.open_zarr(
                    os.path.join(
                        output_path_percentile, "thresholded_verification.zarr"
                    )
                ).load()
                thresholded_ensemble_mean = xarray.open_zarr(
                    os.path.join(
                        output_path_percentile, "thresholded_ensemble_mean.zarr"
                    )
                ).load()

                #logger.info("Calculating contingency table")
                #deterministic_ds = thresholded_ds.isel(ensemble=0)
                #con = xskillscore.Contingency(
                #    deterministic_ds.sel(lat=slice(80,-80)),
                #    thresholded_verification.sel(lat=slice(80,-80)),
                #    np.array([0, 0.5, 1]),
                #    np.array([0, 0.5, 1]),
                #    ("lat", "lon"),
                #)
                #save_dataset(
                #    con.equit_threat_score(),
                #    os.path.join(output_path_percentile, "ets"),
                #)

                #edi = compute_edi(con)
                #save_dataset(edi, os.path.join(output_path_percentile, "edi"))
                #save_dataset(
                #    con.correct_negatives(),
                #    os.path.join(output_path_percentile, "correct_negatives"),
                #)
                #save_dataset(con.hits(), os.path.join(output_path_percentile, "hits"))
                #save_dataset(
                #    con.misses(), os.path.join(output_path_percentile, "misses")
                #)
                #save_dataset(
                #    con.false_alarms(),
                #    os.path.join(output_path_percentile, "false_alarms"),
                #)

                ## Deterministic LogScore
                #logger.info("Scoring deterministic log score")
                #deterministic_log_score = compute_log_score(
                #    thresholded_verification, deterministic_ds, percentile
                #)
                #deterministic_log_score.attrs = ds.attrs
                #save_dataset(
                #    deterministic_log_score, os.path.join(output_path_percentile, "deterministic_log_score")
                #)

                logger.info("Scoring twMAE")
                twMAE = np.abs(
                    xarray.where(verification < threshold, threshold, verification) -
                    xarray.where(ds.isel(ensemble=0) < threshold, threshold, ds.isel(ensemble=0))
                )
                twMAE.attrs = ds.attrs
                save_dataset(twMAE, os.path.join(output_path_percentile, f"twMAE"))

                # threshold_weighted CRPS
                logger.info("Scoring twCRPS")
                twCRPS = xskillscore.crps_ensemble(
                    xarray.where(verification < threshold, threshold, verification),
                    xarray.where(ds < threshold, threshold, ds).chunk(
                        dict(ensemble=-1)
                    ),
                    issorted=False,
                    member_dim="ensemble",
                    dim=(),
                    keep_attrs=True,
                )
                twCRPS.attrs = ds.attrs
                save_dataset(twCRPS, os.path.join(output_path_percentile, f"twCRPS"))

                logger.info("Scoring reliability")
                bin_edges = np.linspace(0, 1, 11)
                reliability = xskillscore.reliability(
                    thresholded_verification,
                    thresholded_ensemble_mean,
                    probability_bin_edges=bin_edges,
                    dim=("lat", "lon"),
                )
                reliability.attrs = ds.attrs
                reliability["bin_edges"] = bin_edges
                save_dataset(
                    reliability, os.path.join(output_path_percentile, f"reliability")
                )
                
                # Brier Score Decomposition
                logger.info("Scoring brier decomposition")
                bin_centers = np.linspace(0.05, 0.95, 10)
                for var in list(ds.keys()):
                    var_TEM = thresholded_ensemble_mean[var] # variable thresholded ensemble mean
                    sharpness = xhistogram.xarray.histogram(var_TEM, bins=[bin_edges], dim=["lat", "lon"])
                    sharpness.attrs["bin_centers"] = bin_centers
                    brier_ds = compute_brier_score_decomposition(
                        sharpness,
                        reliability[var],
                        percentile,
                        var,
                    )
                    brier_score_raw = xskillscore.brier_score(
                        thresholded_verification[var],
                        var_TEM,
                    )
                    brier_ds["brier_score_raw"] = brier_score_raw
                    save_dataset(
                        sharpness.to_dataset(),
                        os.path.join(output_path_percentile, f"{var}_sharpness"),
                    )
                    save_dataset(
                        brier_ds,
                        os.path.join(output_path_percentile, f"{var}_brier_scoring"),
                    )
                    

                # ROC
                logger.info("Scoring roc")
                for var in list(ds.keys()):
                    roc = xskillscore.roc(
                        thresholded_verification[var],
                        thresholded_ensemble_mean[var],
                        dim=("lat", "lon"),
                        return_results="all_as_metric_dim",
                    )
                    roc.attrs = ds.attrs
                    save_dataset(
                        roc.to_dataset(),
                        os.path.join(output_path_percentile, f"{var}_roc"),
                    )
                

                # LogScore
                log_score = compute_log_score(
                    thresholded_verification, thresholded_ensemble_mean, percentile
                )
                log_score.attrs = ds.attrs
                save_dataset(
                    log_score, os.path.join(output_path_percentile, "log_score")
                )

                if efi_path is None:
                    continue
                # EFI
                efi = compute_efi(ds, date_obj, efi_path)
                efi.attrs = ds.attrs
                save_dataset(efi, os.path.join(output_path_percentile, "efi"))

                # EFI ROC
                # ECMWF calculates EFI over daily extremes always
                daily_threshold = set_threshold(
                    list(ds.keys()),
                    date_obj,
                    percentile,
                    extreme_scoring,
                    ds['lat'],
                    window='D',
                )
                daily_thresholded_verification = xarray.where(
                    verification > daily_threshold, 1, 0
                ).sel(time=efi["time"])
                save_dataset(
                    daily_thresholded_verification,
                    os.path.join(
                        output_path_percentile, "daily_thresholded_verification"
                    ),
                )
                logger.info("Scoring efi roc")
                efi = efi.load()
                for var in list(ds.keys()):
                    efi_roc = xskillscore.roc(
                        daily_thresholded_verification[var],
                        xarray.where(efi[var] < 0, 0, efi[var]),
                        bin_edges=np.linspace(0, 1, 11),
                        dim=("lat", "lon"),
                        return_results="all_as_metric_dim",
                    )
                    efi_roc.attrs = ds.attrs
                    save_dataset(
                        efi_roc.to_dataset(),
                        os.path.join(output_path_percentile, f"{var}_efi_roc"),
                    )

        if save_ensemble:
            logger.info("Saving ensemble")
            save_dataset(ds, os.path.join(output_path, "ensemble"))

    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from distributed import Client

    #Client(n_workers=2, memory_limit='480GB')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="full path to the ensemble simulation directory for ML weather",
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
    parser.add_argument(
        "--ifs",
        action="store_true",
        help="whether or not the data being validated is from IFS",
    )
    parser.add_argument(
        "--extreme_scoring",
        default=None,
        type=str,
        help="string containing path to location of extreme scoring dependencies",
    )
    parser.add_argument(
        "--efi_path",
        default=None,
        type=str,
        help="string containing path to location of CDFs necessary for EFI",
    )
    parser.add_argument(
        "--precip",
        action="store_true",
        help="whether or not the data being validated is precip",
    )
    from timeit import default_timer

    start = default_timer()

    args = parser.parse_args()

    if args.efi_path is not None:
        assert (
            args.extreme_scoring is not None
        ), "--extreme_scoring must be set if --efi_path is set"
    main(
        args.input_path,
        args.output_path,
        args.time_averaging_window,
        args.score,
        args.save_ensemble,
        args.ifs,
        args.extreme_scoring,
        args.efi_path,
        args.precip
    )
    logger.info("Script took {}".format(default_timer() - start))
