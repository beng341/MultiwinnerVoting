#!/bin/bash

# Function to check if all network files exist for a given configuration
check_networks_exist() {
    local n_voters=$1
    local m=$2
    local num_winners=$3
    local pref_dist=$4
    
    for idx in {0..19}; do
        network_file="aaai/results/trained_networks/trained_networks/NN-num_voters=${n_voters}-m=${m}-num_winners=${num_winners}-pref_dist=${pref_dist}-axioms=reduced-features=bcr-loss=L1Loss()-idx=${idx}-.pt"
        if [ ! -f "$network_file" ]; then
            return 1
        fi
    done
    return 0
}

# Function to check if evaluation result exists
check_eval_exists() {
    local n_voters=$1
    local m=$2
    local num_winners=$3
    local pref_dist=$4
    
    eval_file="aaai/results/results/axiom_violation_results-n_profiles=25000-num_voters=${n_voters}-m=${m}-k=${num_winners}-pref_dist=${pref_dist}-axioms=reduced.csv"
    if [ -f "$eval_file" ]; then
        return 0
    fi
    return 1
}

# Process each training job
for train_job in aaai/cc_jobs/train_jobs/cc_job_train_*.sh; do
    # Extract parameters from filename
    filename=$(basename "$train_job")
    
    # Extract n_voters, m, k (num_winners), and pref_dist
    n_voters=$(echo "$filename" | grep -o "n_voters=[0-9]*" | cut -d'=' -f2)
    m=$(echo "$filename" | grep -o "m=[0-9]*" | cut -d'=' -f2)
    num_winners=$(echo "$filename" | grep -o "k=[0-9]*" | cut -d'=' -f2)
    
    # Extract pref_dist (this is more complex due to possible special characters)
    pref_dist=$(echo "$filename" | grep -o "pref_dist='[^']*'" | sed "s/pref_dist='\([^']*\)'/\1/")
    
    # Skip if we couldn't extract all parameters
    if [ -z "$n_voters" ] || [ -z "$m" ] || [ -z "$num_winners" ] || [ -z "$pref_dist" ]; then
        echo "Skipping $filename - couldn't extract all parameters"
        continue
    fi
    
    # Check if all network files exist
    if check_networks_exist "$n_voters" "$m" "$num_winners" "$pref_dist"; then
        # Check if evaluation result doesn't exist
        if ! check_eval_exists "$n_voters" "$m" "$num_winners" "$pref_dist"; then
            echo "Submitting eval job for: n_voters=$n_voters m=$m k=$num_winners pref_dist=$pref_dist"
            
            # Find corresponding eval job
            eval_job="aaai/cc_jobs/eval_jobs/cc_job_eval_n_profiles=25000_n_voters=${n_voters}_m=${m}_k=${num_winners}_pref_dist='${pref_dist}'.sh"
            
            if [ -f "$eval_job" ]; then
                sbatch "$eval_job"
                echo "Submitted: $eval_job"
            else
                echo "Warning: Could not find eval job: $eval_job"
            fi
        fi
    fi
done 