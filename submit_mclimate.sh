#!/bin/bash
#SBATCH -N 1
#SBATCH -C 'gpu&hbm80g'
#SBATCH -G 2
#SBATCH -q regular
#SBATCH -J time_collection
#SBATCH --mail-user=amahesh@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 16:00:00
#SBATCH -A m4416
#SBATCH --array 0-36
#SBATCH -o logs/mclimate_time_collection_bred_multicheckpoint-%x-%j_%a.out


srun -N 1 --ntasks-per-node=2 --cpus-per-task=64 --gpus-per-node=2 -u shifter --image=amahesh19/modulus-makani:0.1.0-torch_patch-23.11-multicheckpoint --module=gpu,nccl-2.18 bash -c "source set_74ch_vars.sh; python -m earth2mip.time_collection /pscratch/sd/a/amahesh/hens/mclimate_prod/  --shard $SLURM_ARRAY_TASK_ID --n-shards $SLURM_ARRAY_TASK_COUNT --select-six-random-checkpoints"

