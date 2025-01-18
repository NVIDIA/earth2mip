"""
This script calculates the information gain at each grid cell


srun --mpi=pmi2 -N 4 -u --ntasks-per-node=23 --cpu-bind=cores -c 10 shifter --module none --image=registry.nersc.gov/dasrepo/pharring/deepspeed-pytorch:24.04 python -u paralleltrials_gain_gridcell.py --variable t2m --slurm_array_id 0 --slurm_array_size 92
"""

from mpi4py import MPI
import h5py as h5
import argparse
import pandas as pd
from datetime import timedelta
import numpy as np
from timeit import default_timer

def save_statistic(statistic, reduced_forecast, initial_time_idx, trial_idx, ensemble_size_idx, fout):
    start = default_timer()
    #2000 trials for each bootstrap
    extended_shape = [92, 2, 2000] + list(reduced_forecast.shape)
    if f"{args.variable}_{statistic}" not in fout:
        ds = fout.create_dataset(f"{args.variable}_{statistic}", shape=extended_shape)
    else:
        ds = fout[f"{args.variable}_{statistic}"]
    ds[initial_time_idx, ensemble_size_idx, trial_idx] = reduced_forecast
    #print(f"Time taken to save {statistic} is {default_timer() - start}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Stats on HENS output files")
    parser.add_argument("--variable", type=str, required=True, help="Variable to calculate stats on")
    parser.add_argument("--slurm_array_id", type=int, required=True, help="Slurm array id")
    parser.add_argument("--slurm_array_size", type=int, required=True, help="Slurm array size")
    args = parser.parse_args()


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Rank: {rank}, Size: {size}")

    np.random.seed(rank)

    dates = pd.date_range(start="2023-06-01", end="2023-08-31", freq="D")

    time_subset = np.asarray([16,17,18,19,28,29,30,31,40,41,42,43]) * pd.Timedelta('6H')

    # Data dimensions are (initial_time, lead_time, lat, ensemble, lon)

    # Use lead time idx 8 --> this corresponds to 10 day sin the subsampled lead time output
    for lead_time_idx in [8]:
        for initial_time_idx, initial_time in enumerate(dates):
            if initial_time_idx % args.slurm_array_size != args.slurm_array_id:
                continue
            fin = h5.File(f"/pscratch/sd/a/amahesh/hens_h5/{args.variable}_{initial_time:%Y%m%d}_lead-04-07-10.h5", 'r',
                        )
            assert fin[args.variable].shape[0] == 1, "Expected 1 initial time per file"
            assert fin[args.variable].shape == (1, 12, 721, 7424, 1440), "Expected shape of (1, 12, 721, 7424, 1440) as the dimension shape"
            
            start = default_timer()
            fout = h5.File(f"/pscratch/sd/a/amahesh/hens_h5/paralleltrials_gain_grid_cell/gain_{args.variable}_{initial_time:%Y%m%d}_reduced-lead-10.h5", 'a',
                           driver='mpio', comm=comm)
            start = default_timer()
            forecast = fin[args.variable][0, lead_time_idx, :, :, :]
            print("Read time is ", default_timer() - start)
            assert forecast.shape == (721, 7424, 1440), "Expected shape of (721, 7424, 1440) as the forecast shape"

            #Calculate stats on a bootstrapped ensemble
            gains_per_initial_time = []
            for ensemble_size_idx, ensemble_size in enumerate([7424, 50]):
                gains = []
                start = default_timer()
                for trial in range(2000):
                    if trial % size != rank:
                        continue
                    #Clear the ensemble to save memory
                    ensemble = None

                    ensemble = forecast[:, np.random.choice(7424, ensemble_size, replace=True)]
                    ensemble_max = ensemble.max(axis=1)
                    ensemble_min = ensemble.min(axis=1)
                    ensemble_mean = ensemble.mean(axis=1)
                    ensemble_std = ensemble.std(axis=1)

                    ens_max_zscore = (ensemble_max - ensemble_mean) / ensemble_std
                    ens_min_zscore = (ensemble_mean - ensemble_min) / ensemble_std

                    gain = np.maximum(ens_max_zscore, ens_min_zscore)
                    save_statistic("gain", gain, initial_time_idx, trial, ensemble_size_idx, fout)
                    print(f"Trial {trial} done")
                    #gains.append(gain)

                print(f"Time taken to bootstrap ensemble size {ensemble_size} is {default_timer() - start}")

                #gains = np.stack(gains, axis=0)
                #gains_per_initial_time.append(gains.mean(axis=0))
                
                #print(f"Ensemble size {ensemble_size} done")
            #gains_per_initial_time = np.stack(gains_per_initial_time, axis=0)
            #save_statistic("gain", gains_per_initial_time, initial_time_idx, fout)

            fin.close()
            fout.close()

                    

                    


