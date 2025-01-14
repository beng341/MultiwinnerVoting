#!/bin/bash

# Count the number of complete experiment results for each combination of parameters
# Use by piping output into a csv file
# > sh count_complete_evaluation_results.sh evaluation_results_fixed_fm >> complete_evaluation_result_counts.csv

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Directory containing the files
dir="$1"

# Parameter values
M_values=(5 6 7)
K_values=(1 2 3 4 5 6)
DIST_values=("stratification__args__weight=0.5" "URN-R" "IC" "IAC" "identity" "MALLOWS-RELPHI-R" "single_peaked_conitzer" "single_peaked_walsh" "euclidean__args__dimensions=3_-_space=gaussian_ball" "euclidean__args__dimensions=10_-_space=gaussian_ball" "euclidean__args__dimensions=3_-_space=uniform_ball" "euclidean__args__dimensions=10_-_space=uniform_ball" "euclidean__args__dimensions=3_-_space=gaussian_cube" "euclidean__args__dimensions=10_-_space=gaussian_cube" "euclidean__args__dimensions=3_-_space=uniform_cube" "euclidean__args__dimensions=10_-_space=uniform_cube" "mixed")

# Output header
echo "M,K,DIST,Count"

# Loop over all combinations of M, K, and DIST
for M in "${M_values[@]}"; do
  for K in "${K_values[@]}"; do
    if (( M > K )); then
      for DIST in "${DIST_values[@]}"; do
        # Pattern to match files
        pattern="axiom_violation_results-n_profiles=25000-num_voters=50-m=${M}-k=${K}-pref_dist=${DIST}-axioms=all.csv"

        # Count matching files in the directory
        count=$(find "$dir" -type f -name "$pattern" | wc -l)

        # Print the result
        echo "$M,$K,$DIST,$count"
      done
    fi
  done
done