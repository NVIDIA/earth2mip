#!/bin/bash

# Define the command template
command_template="shifter --image=amahesh19/modulus-makani:0.1.0-torch_patch-23.11-multicheckpoint --module=gpu,nccl-2.18 --env PYTHONUSERBASE=/global/homes/a/amahesh/gcsfs_usrbase/ bash -c \"source set_74ch_vars.sh; python -u -m earth2mip.score_extreme_diagnostics --output_path \$SCRATCH/ifs_extreme_scores_analysis/%s/ --input_path '%s' --extreme_scoring /pscratch/sd/a/amahesh/hens/thresholds/ --ifs\""

# Loop through each date
for date in "2018-01-02T00:00:00" "2018-01-09T00:00:00" "2018-01-16T00:00:00" "2018-01-23T00:00:00" "2018-01-30T00:00:00" "2018-02-06T00:00:00" "2018-02-13T00:00:00" "2018-02-20T00:00:00" "2018-02-27T00:00:00" "2018-03-06T00:00:00" "2018-03-13T00:00:00" "2018-03-20T00:00:00" "2018-03-27T00:00:00" "2018-04-03T00:00:00" "2018-04-10T00:00:00" "2018-04-17T00:00:00" "2018-04-24T00:00:00" "2018-05-01T00:00:00" "2018-05-08T00:00:00" "2018-05-15T00:00:00" "2018-05-22T00:00:00" "2018-05-29T00:00:00" "2018-06-05T00:00:00" "2018-06-12T00:00:00" "2018-06-19T00:00:00" "2018-06-26T00:00:00" "2018-07-03T00:00:00" "2018-07-10T00:00:00" "2018-07-17T00:00:00" "2018-07-24T00:00:00" "2018-07-31T00:00:00" "2018-08-07T00:00:00" "2018-08-14T00:00:00" "2018-08-21T00:00:00" "2018-08-28T00:00:00" "2018-09-04T00:00:00"
do
    eval $(printf "$command_template" "$date" "$date")
done

