#!/bin/bash
#SBATCH -C cpu
#SBATCH -A m1517
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --cpus-per-task=64
#SBATCH --time 2:10:00
#SBATCH --job-name=mclimate-percentiles
#SBATCH --output=/pscratch/sd/a/amahesh/mclimate-percentile_job_logs/%j_%A_%a.out
#SBATCH --error=/pscratch/sd/a/amahesh/mclimate-percentile_job_logs/%A_%a_error.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=amahesh@lbl.gov
#SBATCH --image=docker:amahesh19/xarray-perlmutter:latest
#SBATCH --array=0-90

# Define the output directory
output_path="/pscratch/sd/a/amahesh/hens/mclimate_prod/daily_mclimate_percentiles_extended/"
mclimate_path="/pscratch/sd/a/amahesh/hens/mclimate_prod/"

echo "Starting scheduler..."

scheduler_file=$output_path/scheduler_file_$SLURM_ARRAY_TASK_ID.json
rm -f $scheduler_file

image=amahesh19/xarray-perlmutter:latest

#start scheduler
DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
shifter --image=$image dask-scheduler \
    --scheduler-file $scheduler_file --interface hsn0 &

dask_pid=$!

# Wait for the scheduler to start
sleep 5
until [ -f $scheduler_file ]
do
     sleep 5
done

echo "Starting workers"

DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=3600s \
DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP=3600s \
srun shifter --image=$image dask-worker \
    --scheduler-file $scheduler_file \
    --nworkers 1 --interface hsn0 &


sleep 10

echo "starting python script"

shifter python3 -u calculate_mclimate_percentiles.py --task_id "$SLURM_ARRAY_TASK_ID" --output_path $output_path --mclimate_path $mclimate_path

echo "Killing scheduler"
kill -9 $dask_pid
