#!/bin/bash
#SBATCH -N 64
#SBATCH -C 'gpu&hbm80g'
#SBATCH -q regular
#SBATCH -J HENS_summer23_%a
#SBATCH --mail-user=amahesh@lbl.gov,wdcollins@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 01:30:00
#SBATCH -A m4416
#SBATCH -o summer23_logs/hens-%j_%A_%a.out
#SBATCH --array=0-1%2
#SBATCH --mail-type=ARRAY_TASKS 

# Define the start and end dates for summer 2023
start_date="2023-06-01T00:00:00Z"
end_date="2023-08-31T00:00:00Z"

# Define the function to generate the date
generate_date() {
    local current_date=$(date -u -d "$start_date + $1 days" "+%Y%m%dT000000")
    echo "$current_date"
}

# Select the current date based on the SLURM_ARRAY_TASK_ID
current_date=$(generate_date $SLURM_ARRAY_TASK_ID)

# Your existing srun command with the selected date
srun -N 64 --ntasks-per-node=4 --cpus-per-task=32 --gpus-per-node=4 -u shifter --image=amahesh19/modulus-makani:0.1.0-torch_patch-23.11-multicheckpoint --module=gpu,nccl-2.18 bash -c "source set_74ch_vars.sh; python -m earth2mip.inference_ensemble summer23_configs/config_$current_date.json"

source deactivate
module load python
conda activate cds_env

# Create the JSON string
# Create the JSON string using jq with current_date
# Create the JSON string using jq with separate args for transfer_label and delete_label
json_string=$(jq -n --arg sd "/pscratch/sd/a/amahesh/hens/HENS_summer23_$current_date/" \
                     --arg dd "/global/cfs/cdirs/m1517/cascade/amahesh/hens/" \
                     --arg fd "$current_date" \
                     --arg transfer_label "Transfer for Generic Move from scratch to CFS $current_date and Array Task ID $SLURM_ARRAY_TASK_ID" \
                     --arg delete_label "Delete after Transfer for Generic Move from scratch to CFS $current_date and Array Task ID $SLURM_ARRAY_TASK_ID" \
                     '{"source": {"id": "6bdc7956-fc0f-4ad2-989c-7aa5ee643a79", "path": $sd},
                      "destination": {"id": "9d6d994a-6d04-11e5-ba46-22000b92c6ec", "path": $dd},
                      "transfer_label": $transfer_label,
                      "delete_label": $delete_label}')


echo $json_string
globus flows start f37e5766-7b3c-4c02-92ee-e6aacd8f4cb8 --input "$json_string" --label "HENS_summer23_$current_date"

