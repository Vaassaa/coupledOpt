#!/bin/bash

# --- CONFIGURATION ---
# Add or remove directory names here
TARGETS=("beech" "spruce")
ARCH_DIR="$(pwd)/arch"
BIN_PATH="bin/drutes" # Absolute path to the binary

# --- THE SIMULATION FUNCTION ---
# This is where the work happens. Each parallel process runs this.
run_simulation() {
    local species=$1
    local arch_dir=$2
    local binary=$3
    local run_path="${arch_dir}/${species}/drutes_run_best"

    echo "[$(date +%T)] Starting $species..."

    # 1. Enter the specific run directory so it can find drutes.conf
    if cd "$run_path" 2>/dev/null; then
        
        # 2. Execute the binary (using absolute path)
        # We redirect output to a log file inside the drutes_run folder
        "$binary" > simulation.log 2>&1
        
        # 3. Check if it succeeded
        if [ $? -eq 0 ]; then
            echo "[$(date +%T)] SUCCESS: $species finished."
        else
            echo "[$(date +%T)] ERROR: $species failed. Check $run_path/simulation.log" >&2
        fi
    else
        echo "[$(date +%T)] CRITICAL: Directory not found: $run_path" >&2
    fi
}

# Export the function and variables so GNU Parallel can see them
export -f run_simulation

# --- EXECUTION ---
echo "Launching simulations in parallel..."

# {} is the species name from the array
# -j 2 runs two at a time
parallel --line-buffer -j 2 run_simulation {} "$ARCH_DIR" "$BIN_PATH" ::: "${TARGETS[@]}"

echo "All jobs dispatched."
