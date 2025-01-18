#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 2
#SBATCH -q regular
#SBATCH -J smallckpt_time_collection
#SBATCH --mail-user=amahesh@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 1:30:00
#SBATCH -A m4416
#SBATCH --array 2-3
#SBATCH -o logs/smallckpt_time_collection_bred_multicheckpoint-%x-%j_%a.out
#SBATCH --image=amahesh19/modulus-makani:0.1.0-torch_patch-23.11-multicheckpoint

srun -N 1 --ntasks-per-node=2 --cpus-per-task=32 --gpus-per-node=2 -u shifter --module=gpu,nccl-2.18 bash -c "source set_74ch_vars.sh; python -m earth2mip.time_collection /pscratch/sd/a/amahesh/hens/time_collection/nopert_33multicheckpoint_pert0p35_k1i3_500km_oppositepert_detfix_nodpr_timeevolve_hemisphererescale_target48_20minus20_newseed_repeat_qperturbfix_rankhist_qmin0/ --shard $SLURM_ARRAY_TASK_ID --n-shards $SLURM_ARRAY_TASK_COUNT"


