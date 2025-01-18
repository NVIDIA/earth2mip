import numpy as np
import xarray
from xhistogram.xarray import histogram
import earth2mip.time


import logging
logger = logging.getLogger(__file__)


def set_threshold(variables, times, percentile, extreme_scoring, align_ds, window=""):
    """
    loads the appropriate threshold (at a given percentile value and
    for a given month)

    (These percentiles need to be precomputed.)

    inputs
    ------

         variables   : the variables for which to calculate the
                           threshold
         initial_condition : the datetime of the initial condition
         percentile        : the percentile threshold
         extreme_scoring   : the path of where the percentile data is
                             stored
    """
    if percentile != '99p9':
        percentile_str = str(int(percentile * 100))
    else:
        percentile_str = percentile
    assert set(variables) == set(
        ["t2m"] #, "tcwv", "wind_speed10m")
    ), "Extreme event validation is only implemented for t2m. These variables are included: {}".format(variables)
    if window == 'D':
        print("Loading the {}th percentile of the daily mean temperatures".format(percentile_str))
        #Load the Xth percentile of daily mean temperatures
        #initial_condition = earth2mip.time.convert_to_datetime(times[0])
        #ds = xarray.open_dataset(
        #    "{}/percentile_{}/{}/{:02d}/wind_tcwv_t2m_percentile_{}.zarr".format(
        #        "/global/cfs/cdirs/m4416/",
        #        percentile_str,
        #        window,
        #        initial_condition.month,
        #        percentile_str,
        #    )
        #)
        #print(initial_condition.month)
        #return ds[variables].load()
        thresholds = []
        for curr_time in times:
            curr_time_dt = earth2mip.time.convert_to_datetime(curr_time)
            ds = xarray.open_zarr(
                "{}/t2m_percentile{}_{:02d}_dailymean.zarr".format(
                    extreme_scoring.replace('thresholds', 'thresholds_daily'),
                    percentile_str,
                    curr_time_dt.month,
                )
            )
            thresholds.append(ds)
        
        retval = xarray.concat(thresholds, dim=times).rename({'latitude' : 'lat', 'longitude' : 'lon',
                                                            'VAR_2T' : 't2m'})[['t2m']]
        _, retval = xarray.align(align_ds, retval)
        return retval.chunk({'time': 1,
               'lat': 721,
               'lon': 1440})

    else:
        #Load the Xth percentile of each month of each day
        print("Loading the {}th percentile of each month+hour".format(percentile_str))
        thresholds = []
        for curr_time in times:
            curr_time_dt = earth2mip.time.convert_to_datetime(curr_time)
            ds = xarray.open_zarr(
                "{}/t2m_percentile{}_{:02d}_{:02d}".format(
                    extreme_scoring,
                    percentile_str,
                    curr_time_dt.month,
                    curr_time_dt.hour,
                )
            )
            thresholds.append(ds)
        
        retval = xarray.concat(thresholds, dim=times).rename({'latitude' : 'lat', 'longitude' : 'lon',
                                                            'VAR_2T' : 't2m'})[['t2m']]
        _, retval = xarray.align(align_ds, retval)
        return retval.chunk({'time': 1,
               'lat': 721,
               'lon': 1440})


def threshold_at_extremes(ds, threshold):
    """
    inputs
    ------

           ds        : the dataset to threshold (e.g. truth or preds)
           threshold : the thresholds above which values are 1 (
                       below them the values are 0)
    """
    binarized = xarray.where(ds > threshold, 1, 0)
    return binarized


def compute_reliability(pred_mean, target_binary, lead_times, n_bins=11):
    """
    method to compute reliability diagram manually
    """
    time_subsetted_pred_mean = pred_mean.sel(time=lead_times)
    time_subsetted_target_binary = target_binary.sel(time=lead_times)

    num_predictions = np.zeros(n_bins)
    num_true_events = np.zeros(n_bins)

    rounded_pred_mean = time_subsetted_pred_mean.round(1)

    # a bin corresponds to a probability interval, ie 0-5%, 5-15%, ..., 95-100%
    # bin size changes according to n_bins
    bins = np.linspace(0, 1, n_bins)
    for i, bin_prob in enumerate(bins):
        bin_mask = xarray.where(rounded_pred_mean == bin_prob, True, False)
        num_predictions[i] = bin_mask.sum()  # number of predictions in this bin

        bin_events = time_subsetted_target_binary[bin_mask]
        num_true_events[i] = bin_events.sum()  # true number of events in this bin

    num_predictions_da = xarray.DataArray(
        num_predictions, dim=("bin"), coords={"bin": bins}
    )
    num_true_events_da = xarray.DataArray(
        num_true_events, dim=("bin"), coords={"bin": bins}
    )
    reliability_ds = xarray.merge((num_predictions_da, num_true_events_da))

    return reliability_ds

def compute_brier_score_decomposition(sharpness, reliability, percentile, var):
    """
    Brier score decomposition (3 part): reliability, resolution, uncertainty. BS = rel - res + unc
    """
    if percentile == '99p9':
        percentile = 0.999
    n_k  = sharpness.rename({f"{var}_bin": "forecast_probability"} )
    N    = n_k.sum()
    o_k  = reliability
    obar = 1 - percentile
    f_k  = sharpness.attrs["bin_centers"]
    
    rel = (n_k * np.square(f_k - o_k)).sum() / N
    res = (n_k * np.square(o_k - obar)).sum() / N
    unc = obar * (1 - obar)
    
    bs_decomp = rel - res + unc
    
    ds = xarray.Dataset({
        'reliability': rel,
        'resolution': res,
        'uncertainty': unc,
        'brier_score_decomp': bs_decomp,
    })
    ds.attrs["bin_centers"] = f_k
        
    return ds
    


def _compute_log_score(truth, preds):
    """
    computes the binary cross entropy elementwise of truth
    and preds
    """
    return -(truth * np.log(preds + 1e-9) + (1 - truth) * np.log(1 - preds + 1e-9))


def compute_log_score(truth, preds, percentile_threshold):
    """
    computes the mean binary cross entropy across all grid cells,
    weighted by the cos of latitude

    inputs
    ------
        truth  : the thresholded values of the ground truth
        preds  : the thresholded ensemble mean of the forecast
        percentile_threshold   : the percentile threshold used on
                             truth and preds.  the reference
                             climatological forecast would predict
                             this value for all grid cells
    """
    log_scores = _compute_log_score(truth, preds)
    return log_scores
    #ref_log_score = _compute_log_score(truth, 1 - percentile_threshold)
    #log_scores = 1 - (log_scores / ref_log_score)

    #weights = np.cos(np.deg2rad(preds.lat))
    #weighted_log_score = log_scores.weighted(weights).mean(dim=("lat", "lon"))
    #return weighted_log_score


def compute_efi(preds, initial_condition, efi_path, mode, ifs=False):
    """
    Compute the extreme forecast index: the distance between
    the ensemble distribution at the current lead time and
    the CDF for that day of the model climate

    More info: https://confluence.ecmwf.int/display/FUG/
            Section+8.1.9.2+Extreme+Forecast+Index+-+EFI

    inputs
    -----
        preds              : the ensemble predictions (not thresholded)
        initial condition  : the datetime of the initial condition
        path               : the path to the location of the precomputed
                             model climate percentiles
        mode               : mean or max


    """
    if ifs:
        print("Loading in IFS EFI")
        ifs_efi = xarray.open_dataset("/pscratch/sd/a/amahesh/ifs_efi/grid_025/efi_{:04d}_{:02d}_01.grib".format(initial_condition.year, initial_condition.month),
                         filter_by_keys={'paramId': 132167})
        ifs_efi = ifs_efi.sel(time=initial_condition)
        ifs_efi['step'] = ifs_efi['time'] + ifs_efi['step'] - np.timedelta64(1, 'D')
        ifs_efi = ifs_efi.rename({'time' : 'initial_time',
                'step': 'time',
                't2i' : 't2m',
                'latitude' : 'lat',
                'longitude' : 'lon'})
        return ifs_efi
    print("Calculating SFNO EFI")
    # CDF of the model climate must be precomputed.  See
    # fcn_mip/workflows/mclimate_statistics/
    cdf = xarray.open_zarr(
        "{}/MClimate_{}_Percentiles.2023-{:02d}-{:02d}.zarr".format(
            efi_path, mode, initial_condition.month, initial_condition.day
        )
    )

    preds_daily = preds.resample(time="D").mean()
    F_p = (preds_daily <= cdf).mean("ensemble")
    efi = (
        2
        / np.pi
        * (
            (cdf["quantile"] - F_p) / np.sqrt(cdf["quantile"] * (1 - cdf["quantile"]))
        ).integrate("quantile")
    )
    return efi


def compute_edi(con):
    """
    inputs
    ------
        con : xskillscore Contingency table
    outputs
    -------
        Extremal Dependence Index, a score based on the hit rate
        and false alarm rate

        more info: equation 1 of https://doi.org/10.1175/WAF-D-10-05030.1

    """

    H = con.hit_rate()
    F = con.false_alarm_rate()

    return (np.log(F) - np.log(H)) / (np.log(F) + np.log(H))
