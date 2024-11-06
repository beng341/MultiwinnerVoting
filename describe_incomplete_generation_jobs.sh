#!/bin/bash

# Define variables for parameters
M_VALUES=(5 6 7)
K_VALUES=(1 2 3 4 5 6)
DIST_VALUES=(
    "stratification__args__weight=0.5"
    "URN-R"
    "IC"
    "IAC"
    "identity"
    "MALLOWS-RELPHI-R"
    "single_peaked_conitzer"
    "single_peaked_walsh"
    "euclidean__args__dimensions=3_-_space=gaussian_ball"
    "euclidean__args__dimensions=10_-_space=gaussian_ball"
    "euclidean__args__dimensions=3_-_space=uniform_ball"
    "euclidean__args__dimensions=10_-_space=uniform_ball"
    "euclidean__args__dimensions=3_-_space=gaussian_cube"
    "euclidean__args__dimensions=10_-_space=gaussian_cube"
    "euclidean__args__dimensions=3_-_space=uniform_cube"
    "euclidean__args__dimensions=10_-_space=uniform_cube"
)

# Define the output CSV file
output_csv="incomplete_job_descriptions.csv"

# Write header to the CSV file
# echo "filename,train_lines,test_lines,total_lines_fraction" > "$output_csv"
echo "m,k,preference_distribution,train_lines_total,test_lines_total,total_lines_fraction,hours_given_job" >> "$output_csv"      

# Function to count lines in a file (returns 0 if the file does not exist)
count_lines() {
    if [[ -f "$1" ]]; then
        wc -l < "$1"
    else
        echo 0
    fi
}

# Function to calculate hours given to data_generation jobs
calculate_hours() {
    local m=$1
    local k=$2

    # Calculate k ** 0.6
    # local k_pow_0_6=$(echo "$k^0.6" | bc -l)
    local k_pow_0_6=$(awk -v k="$k" 'BEGIN { printf "%.5f", k^0.6 }')
    # local k_pow_0_6=2

    # Function to calculate factorial
    factorial() {
        local n=$1
        local fact=1
        for (( i=1; i<=n; i++ )); do
            fact=$((fact * i))
        done
        echo $fact
    }

    # Calculate binomial coefficient binom(m, k)
    local m_fact=$(factorial $m)
    local k_fact=$(factorial $k)
    local m_minus_k_fact=$(factorial $((m - k)))
    local binom=$(echo "$m_fact / ($k_fact * $m_minus_k_fact)" | bc)

    # Calculate hours
    local hours=$(echo "$k_pow_0_6 * $binom + 1" | bc -l)

    # Output the result (as the function's return value)
    echo "$hours"
}

# base_dir="/scratch/b8armstr/data"
base_dir="data"

# Loop over each combination of M, K, and DIST values
for M in "${M_VALUES[@]}"; do
    for K in "${K_VALUES[@]}"; do
        # Only consider K values less than M
        if (( K < M )); then
            for DIST in "${DIST_VALUES[@]}"; do
                # Define filenames for TRAIN and TEST files


                file_train="${base_dir}/n_profiles=25000-num_voters=50-m=${M}-committee_size=${K}-pref_dist=${DIST}-axioms=all-TRAIN.csv"
                file_test="${base_dir}/n_profiles=25000-num_voters=50-m=${M}-committee_size=${K}-pref_dist=${DIST}-axioms=all-TEST.csv"
                
                # Count lines in each file
                train_lines=$(count_lines "$file_train")
                test_lines=$(count_lines "$file_test")
                
                # Calculate the fraction of total lines (50002 max possible)
                total_lines=$((train_lines + test_lines))
                fraction=$(echo "scale=5; $total_lines / 50002" | bc)

                given_hours=$(calculate_hours $M $K)
                # given_hours=2

                # Write result to CSV
                # echo "$file_train,$train_lines,$test_lines,$fraction" >> "$output_csv"
                echo "$M,$K,$DIST,$train_lines,$test_lines,$fraction,$given_hours" >> "$output_csv"
            done
        fi
    done
done

echo "Line count report saved to $output_csv"