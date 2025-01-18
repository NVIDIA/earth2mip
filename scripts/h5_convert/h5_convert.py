from datetime import datetime, timedelta
import numpy as np
import os
from timeit import default_timer
import argparse
from mpi4py import MPI
import h5py as h5

def create_initial_dates():
    """
    Mimic pd.date_range('2023-06-01', '2023-08-31', freq='D')

    Pandas was not available in the shifter image
    """

    start_date = datetime.strptime('2023-06-01', '%Y-%m-%d')
    end_date = datetime.strptime('2023-08-31', '%Y-%m-%d')

    initial_dates = []
    current_date = start_date
    while current_date <= end_date:
        initial_dates.append(current_date)
        current_date += timedelta(days=1)
    
    return initial_dates

def read_arr(filepath, time_subset, var):
    start_read = default_timer()
    
    with h5.File(filepath, "r") as fin:
        arr = fin['global'][var][:, time_subset, :, :]
    assert arr.shape == (1, len(time_subset), 721, 1440), \
            "Expected [1, 12, 721, 1440]" #Arr shape is ens, lead_time, lat, lon
    retval = arr[0, : , :, :]
    return retval

def write_arr(initial_date_idx, ens_idx, arr, h5_path, dset):
    """
    Write a single dataset to the h5 file

    initial_date_idx        : the index of the initial date
    ens_idx                      : the index of the ensemble member
    arr                          : the array to write
    h5_path                      : the path to the h5 file
    dset                         : the h5 dataset
    """
    start_write = default_timer()
    #print("starting write")
    assert arr.shape == (12, 721, 1440), \
            "Must write an array of size 12, 721, 1440"
    dset[initial_date_idx, :, :, ens_idx, : ] = arr
    if rank // 8 == 0:
        files_per_second = 8 / (default_timer()-start_write)
        print("Files per second", files_per_second)
    #print("completed write in", default_timer()-start_write, "seconds")

def get_base_path(curr_date):
    """
    Check if the data for the current date is on scratch or not

    curr_date: the current date
    """
    if curr_date.day < 14 and curr_date.month == 6:
        return  "/global/cfs/cdirs/m1517/cascade/amahesh/hens/"
    if curr_date.day > 24 and curr_date.month == 8:
        return "/global/cfs/cdirs/m1517/cascade/amahesh/hens/"
    return "/pscratch/sd/a/amahesh/hens_copy/"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HENS data to h5')
    parser.add_argument('--variable', type=str, default='t2m', help='Variable to convert')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"{rank} of {size}")

    initial_dates = create_initial_dates() 

    # Select time_subset: days 4, 7, 10
    time_subset = [16,17,18,19,28,29,30,31,40,41,42,43]

    # There are 736 ranks spread across 92 dates.  There are 8 ranks for each day.
    #The first 8 ranks work for the first day. The next 8 ranks work for the next day, etc.
    which_date_for_rank = rank // 8
    which_ens_for_rank = rank % 8

    group_comm = comm.Split(which_date_for_rank, rank)

    for date_idx, curr_date in enumerate(initial_dates):
        if date_idx == which_date_for_rank:
            BASE_PATH = get_base_path(curr_date)
            print("Rank", rank, "working on", curr_date)
            # Create the dataset
            h5_path = f"/pscratch/sd/a/amahesh/hens_h5/{args.variable}_{curr_date:%Y%m%d}_lead-04-07-10.h5"
            f = h5.File(h5_path, 'a', driver='mpio', comm=group_comm)
            
            # Dataset ordering is initial_time, lead_time, lat, ensemble, lon
            #There is only 1 initial time per file
            dset = f.create_dataset(args.variable, 
                        (1, len(time_subset), 721, 7424, 1440), 
                        dtype=np.float32)
            
            start_ic = default_timer()
            if rank // 8 == 0:
                print("Starting", curr_date, date_idx, "of", len(initial_dates))
            
            for ens in range(7424):
                if ens % 8 == which_ens_for_rank:
                    start_ens = default_timer()
                    arr = read_arr(f"{BASE_PATH}/HENS_summer23_{curr_date:%Y%m%d}T000000/ensemble_out_{ens:05d}_{curr_date:%Y-%m-%d}-00-00-00.nc",
                                time_subset, args.variable)
                    
                    #Replace 0 with date_idx if you are writing to the same file for each date
                    write_arr(0, ens, arr, h5_path, f[args.variable])
                    #print("completed", ens, "of", 7424, "in", default_timer()-start_ens)

            #print("completed", date, idx, "of", len(dates), "in", default_timer()-start_ic)
        
            f.close()
    
    print("finished rank", rank)
    comm.Barrier()
    MPI.Finalize()
