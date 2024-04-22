#!/bin/bash

# Load the contents of sample_config into a variable
sample_config=$(<sample_config)

# Define the start and end dates for summer 2023
start_date="2023-06-01T00:00:00"
end_date="2023-08-31T00:00:00"

# Define the output directory
output_dir="./configs"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Function to generate config file for a given date
generate_config() {
    local new_date="$1"
    local formatted_date=$(date -u -d "$new_date" "+%Y%m%dT000000")
    local start_formatted_date=$(date -u -d "$new_date" "+%Y-%m-%d 00:00:00")
    local output_path="/pscratch/sd/a/amahesh/hens/HENS_summer23_$formatted_date/"
    local config_file="config_${formatted_date}.json"
    local new_config=$(echo "$sample_config" | sed -e "s/\"start_time\": \"2023-06-01 00:00:00\"/\"start_time\": \"$start_formatted_date\"/" -e "s#/pscratch/sd/a/amahesh/hens/HENS_summer23_20230601#$output_path#")
    echo "$new_config" > "$output_dir/$config_file"
}

# Loop through each day from start_date to end_date
current_date=$(date -u -d "$start_date" "+%Y-%m-%d")
end_date=$(date -u -d "$end_date" "+%Y-%m-%d")

while [[ "$current_date" < "$end_date" ]]; do
    generate_config "${current_date}T00:00:00Z"
    current_date=$(date -u -d "$current_date + 1 day" "+%Y-%m-%d")
done

