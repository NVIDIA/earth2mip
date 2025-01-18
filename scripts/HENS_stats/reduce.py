from mpi4py import MPI
import h5py as h5
import xarray as xr
import argparse
import pandas as pd
from datetime import timedelta
import numpy as np
from timeit import default_timer

def load_percentile_threshold(initial_time, lead_time):
    valid_time = initial_time + lead_time
    threshold = xr.open_zarr(f"/pscratch/sd/a/amahesh/hens/thresholds/t2m_percentile99_{valid_time.month:02d}_{valid_time.hour:02d}/")
    threshold = threshold.rename({
        'VAR_2T' : 't2m',
        'latitude' : 'lat',
        'longitude' : 'lon',
    })
    return threshold['t2m'].values[:, None]

def load_observed(initial_time, lead_time):
    valid_time = initial_time + lead_time
    true = xr.open_dataset("/pscratch/sd/p/pharring/74var-6hourly/staging/2023.h5", mode='r')
    dummy = xr.open_dataset("/dvs_ro/cfs/cdirs/m1517/cascade/amahesh/hens/HENS_summer23_20230814T000000/ensemble_out_00001_2023-08-14-00-00-00.nc",
                       group='global')

    true = true.rename({'phony_dim_0' : 'time',
             'phony_dim_2' : 'lat', 
             'phony_dim_3' : 'lon',
             'phony_dim_1' : 'channel'})
    true['lat'] = dummy['lat']
    true['lon'] = dummy['lon']
    true['channel'] = ["u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv", "2d", "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925", "u1000", "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000", "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925", "z1000", "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000", "q50", "q100", "q150", "q200", "q250", "q300", "q400", "q500", "q600", "q700", "q850", "q925", "q1000"]
    true['time'] = pd.date_range("2023-01-01", periods=1460, freq='6H')
    return true['fields'].sel(time=valid_time, channel='t2m').load().values

def save_statistic(statistic, reduced_forecast, initial_time_idx, lead_time_idx, fout):
    start = default_timer()
    extended_shape = [92, 12] + list(reduced_forecast.shape)
    if f"{args.variable}_{statistic}" not in fout:
        ds = fout.create_dataset(f"{args.variable}_{statistic}", shape=extended_shape)
    else:
        ds = fout[f"{args.variable}_{statistic}"]
    ds[initial_time_idx, lead_time_idx] = reduced_forecast
    print(f"Time taken to save {statistic} is {default_timer() - start}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Stats on HENS output files")
    parser.add_argument("--variable", type=str, required=True, help="Variable to calculate stats on")
    parser.add_argument("--slurm_array_id", type=int, required=True, help="Slurm array id")
    parser.add_argument("--slurm_array_size", type=int, required=True, help="Slurm array size")
    args = parser.parse_args()

    np.random.seed(0)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Rank: {rank}, Size: {size}")

    dates = pd.date_range(start="2023-06-01", end="2023-08-31", freq="D")

    time_subset = np.asarray([16,17,18,19,28,29,30,31,40,41,42,43]) * pd.Timedelta('6H')

    # Data dimensions are (initial_time, lead_time, lat, ensemble, lon)

    for lead_time_idx, lead_time in enumerate(time_subset):
        if lead_time_idx % size != rank:
            continue
        for initial_time_idx, initial_time in enumerate(dates):
            if initial_time_idx % args.slurm_array_size != args.slurm_array_id:
                continue
            fin = h5.File(f"/pscratch/sd/a/amahesh/hens_h5/{args.variable}_{initial_time:%Y%m%d}_lead-04-07-10.h5", 'r',
                        driver='mpio', comm=comm)
            assert fin[args.variable].shape[0] == 1, "Expected 1 initial time per file"
            assert fin[args.variable].shape == (1, 12, 721, 7424, 1440), "Expected shape of (1, 12, 721, 7424, 1440) as the dimension shape"
            
            print("Starting fout")
            start = default_timer()
            fout = h5.File(f"/pscratch/sd/a/amahesh/hens_h5/stats/{args.variable}_{initial_time:%Y%m%d}_reduced-lead-04-07-10.h5", 'a', driver='mpio', comm=comm)
            print(f"Time taken to open fout is {default_timer() - start}")
            start = default_timer()
            forecast = fin[args.variable][0, lead_time_idx, :, :, :]
            print("Read time is ", default_timer() - start)
            assert forecast.shape == (721, 7424, 1440), "Expected shape of (721, 7424, 1440) as the forecast shape"

            print("starting saved statistic calc")
            #Calculate general stats over the whole ensemble mean
            save_statistic("mean", forecast.mean(axis=1), initial_time_idx, lead_time_idx, fout)
            save_statistic("max", forecast.max(axis=1), initial_time_idx, lead_time_idx, fout)
            save_statistic("min", forecast.min(axis=1), initial_time_idx, lead_time_idx,fout)
            save_statistic("argmax", forecast.argmax(axis=1), initial_time_idx, lead_time_idx,fout)
            save_statistic("argmin", forecast.argmin(axis=1), initial_time_idx,lead_time_idx, fout)
            save_statistic("var", forecast.var(axis=1), initial_time_idx, lead_time_idx, fout)
            observed = load_observed(initial_time, lead_time)
            ens_mse = (forecast.mean(axis=1) - observed)**2
            save_statistic("ens_mse", ens_mse, initial_time_idx, lead_time_idx, fout)

            #Calculate stats on percentile threshold forecasts
            if args.variable == 't2m':
                percentile_threshold = load_percentile_threshold(initial_time, lead_time)
            else:
                percentile_threshold = None

            #Calculate the number of ensemble members that exceed the observed value
            save_statistic("exceed_observed", (forecast > observed[:, None]).sum(axis=1), initial_time_idx,lead_time_idx, fout)
            save_statistic("below_observed", (forecast < observed[:, None]).sum(axis=1), initial_time_idx,lead_time_idx, fout)

            #Calculate stats on a bootstrapped ensemble
            satisfactory_ens_size, extreme_confidence_interval, extreme_bootstrap_mean, min_mse_interval = [], [], [], []
            min_mse_bootstrap_mean = []
            for ensemble_size in [50, 100, 500, 1000, 5000, 7424]:
                min_mse_list, success, extreme_forecasts = [], [], []
                start = default_timer()
                for trial in range(100):
                    #Clear the ensemble to save memory
                    ensemble = None

                    ensemble = forecast[:, np.random.choice(7424, ensemble_size, replace=True)]
                    ensemble_max = ensemble.max(axis=1)
                    ensemble_min = ensemble.min(axis=1)

                    #Calculate the number of ensemble members that exceed the percentile threshold
                    if percentile_threshold is not None:
                        extreme_forecast_probability = (ensemble > percentile_threshold).mean(axis=1)
                        extreme_forecasts.append(extreme_forecast_probability)
                    
                    min_mse = np.min((ensemble - observed[:, None])**2, axis=1)
                    min_mse_list.append(min_mse)

                    #Determine if the min and max of the ensemble brackets the truth
                    success.append((ensemble_min < observed) & (ensemble_max > observed))
                print(f"Time taken to bootstrap ensemble size {ensemble_size} is {default_timer() - start}")
                
                success = np.stack(success, axis=0)
                extreme_forecasts = np.stack(extreme_forecasts, axis=0)
                min_mse_list = np.stack(min_mse_list, axis=0)

                #If 95% of the trials are success, the ensemble size is satisfactory
                satisfactory_ens_size.append(np.mean(success, axis=0) > 0.95)

                #Calculate the 97.5th and 2.5th percentile of the extreme forecast probabilities
                if percentile_threshold is not None:
                    extreme_confidence_interval.append(np.percentile(extreme_forecasts, [2.5, 97.5], axis=0))
                    extreme_bootstrap_mean.append(extreme_forecasts.mean(axis=0))
                
                min_mse_interval.append(np.percentile(min_mse_list, [2.5, 97.5], axis=0))
                min_mse_bootstrap_mean.append(np.mean(min_mse_list, axis=0))
                print(f"Ensemble size {ensemble_size} done")
            satisfactory_ens_size = np.stack(satisfactory_ens_size, axis=0)
            extreme_confidence_interval = np.stack(extreme_confidence_interval, axis=0)
            min_mse_interval = np.stack(min_mse_interval, axis=0)
            min_mse_bootstrap_mean = np.stack(min_mse_bootstrap_mean, axis=0)
            extreme_bootstrap_mean = np.stack(extreme_bootstrap_mean, axis=0)

            save_statistic("satisfactory_ens_size", satisfactory_ens_size, initial_time_idx,lead_time_idx, fout)
            save_statistic("extreme_confidence_interval", extreme_confidence_interval, initial_time_idx,lead_time_idx, fout)
            save_statistic("extreme_bootstrap_mean", extreme_bootstrap_mean, initial_time_idx, lead_time_idx, fout)
            save_statistic("min_mse_interval", min_mse_interval, initial_time_idx, lead_time_idx,fout)
            save_statistic("min_mse_bootstrap_mean", min_mse_bootstrap_mean, initial_time_idx, lead_time_idx, fout)
            fin.close()
            fout.close()

                    

                    


