#!/bin/bash
#SBATCH -N 4
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J t2m_reconvert
#SBATCH --mail-user=amahesh@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 03:00:00
#SBATCH -A m1517
#SBATCH --job-name=h5_create
#SBATCH --output=logs/t2m_h5_create_%j.log
#SBATCH --image=nersc/pytorch:ngc-22.02-v0 


#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export HDF5_USE_FILE_LOCKING=FALSE

srun -N 4 -u --mpi=pmi2 --module=none -n 736 -c 1 --cpu_bind=threads shifter python h5_convert.py --variable t2m
