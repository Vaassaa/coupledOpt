#!/bin/bash

# Define the working directory path.
work_dir="dataIN/meteo/Campaign_08-09-2024_30-09-2024/"

# Function to get dimensions and check if they are the same.
check_dimensions() {
    local first_dim=""
    for file in "$@"; do
        dim=$(wc -l < "${work_dir}${file}")
        echo "Dimensions of ${file}:"
		echo "$dim"
        # Check if this is the first dimension read or matches previous one
        if [[ ! $first_dim ]]; then
            first_dim=$dim
        elif [[ $first_dim != $dim ]]; then
            echo "ERROR: Dimensions do not match for ${file} "
			echo "($dim)"
            exit 1
        fi
    done
    echo "All files have the same dimensions."
}

# Check *.in files.
echo "Checking regular .in files:"
check_dimensions clouds.in  rh.in  temp.in wind.in

# Check *_interp.in files separately.
echo "Checking _interp.in files (should be different):"
check_dimensions clouds_interp.in rh_interp.in temp_interp.in wind_interp.in rain_freerange.in rain_tree.in solar.in
