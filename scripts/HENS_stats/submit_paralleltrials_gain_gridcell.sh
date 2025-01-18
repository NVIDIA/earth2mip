#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J reduce_t2m
#SBATCH --mail-user=amahesh@lbl.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -t 01:00:00
#SBATCH -A m1517
#SBATCH --job-name=h5_create
#SBATCH --output=logs/parallel_trials_gain_gridcell-%j.log
#SBATCH --image=registry.nersc.gov/dasrepo/pharring/deepspeed-pytorch:24.04
#SBATCH --array=0-91

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export HDF5_USE_FILE_LOCKING=FALSE

module load python
srun --mpi=pmi2 -N 4 -u --ntasks-per-node=4 --cpu-bind=cores -c 64 shifter --module gpu python -u paralleltrials_gain_gridcell.py --variable t2m --slurm_array_id $SLURM_ARRAY_TASK_ID --slurm_array_size $SLURM_ARRAY_TASK_COUNT 
