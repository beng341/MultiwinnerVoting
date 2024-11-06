#!/bin/bash

# Create the output file and add the shebang
output_file="run_incomplete_jobs.sh"
echo "#!/bin/bash" > "$output_file"

# Read the CSV file and generate sbatch commands
awk -F',' '
    BEGIN { OFS="," }
    NR > 1 {  # Skip the header row
        m = $1
        k = $2
        dist = $3
        total_lines_fraction = $6

        # Only create sbatch command if total_lines_fraction is less than 1
        if (total_lines_fraction < 1) {
            printf "sbatch cc_jobs/data_generation/cc_job_n_profiles\\=25000_n_voters\\=50_n_alternative\\=%s_n_winners\\=%s_pref_dist\\=\\'\''%s\\'\''_axioms\\=\\'\''all\\'\''.sh\n", m, k, dist
        }
    }
' incomplete_job_descriptions.csv >> "$output_file"
