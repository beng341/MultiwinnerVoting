#!/bin/bash

folder_path="cc_jobs/test_jobs"

# Loop through all .sh files in the folder
for file in "$folder_path"/*.sh; do
  # Check if the file exists (in case there are no .sh files)
  if [ -e "$file" ]; then
    echo "Submitting: $file"
    # Run sbatch on the .sh file
    sbatch "$file"
  else
    echo "No .sh files found in $folder_path"
    break
  fi
done