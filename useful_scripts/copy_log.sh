#!/bin/bash

# Create the target directory
target_directory="all_log"
mkdir -p "$target_directory"

# Find and copy .log files
find . -type f -name "*.log" | while read -r file; do
    # Get the subdirectory path
    subdir=$(dirname "$file" | sed 's|^\./||')
    
    # Check if the subdirectory contains the capital letter "L"
    if [[ "$subdir" == *L* ]]; then
        # Format the subdirectory path and extract the filename
        formatted_subdir=$(echo "$subdir" | tr '/' '_')
        filename=$(basename "$file")

        # Copy the file to the target directory with the new name
        cp "$file" "$target_directory/${formatted_subdir}_${filename}"
    fi
done

echo "All .log files from subdirectories containing 'L' have been copied to $target_directory."

