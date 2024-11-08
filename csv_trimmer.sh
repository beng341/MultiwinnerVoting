#!/bin/bash

directory="data"

# Check if directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Directory '$directory' not found"
    exit 1
fi

# # First pass: Count and display line numbers for all CSV files
# echo "Current line counts for CSV files:"
# echo "--------------------------------"
# find "$directory" -type f -name "*.csv" | while read -r file; do
#     line_count=$(wc -l < "$file")
#     printf "%-70s %8d lines\n" "$file" "$line_count"
# done

# Second pass: Trim the files
echo -e "\nTrimming files..."
find "$directory" -type f -name "*.csv" | while read -r file; do
    # echo "Processing: $file"

    line_count=$(wc -l < "$file")
    printf "%8d lines %-70s \n" "$line_count" "$file"
    
    # Create a temporary file
    temp_file="${file}.temp"
    
    # Get the header and first 25000 data lines (total 25001 lines)
    head -n 25001 "$file" > "$temp_file"
    
    # Check if operation was successful
    if [ $? -eq 0 ]; then
        # Replace original file with trimmed version
        mv "$temp_file" "$file"
        echo "Successfully trimmed: $file"
    else
        rm -f "$temp_file"
        echo "Error processing: $file"
    fi
done

echo "All CSV files have been processed"