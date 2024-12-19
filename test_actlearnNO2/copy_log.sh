#!/bin/bash

# Create the target directory
target_directory="all_log_new"
mkdir -p "$target_directory"

# Find and copy .xyz files
find . -type f -name "*.log" | while read -r file; do
    # Get the subdirectory path, format it, and extract the filename
    subdir=$(dirname "$file" | sed 's|^\./||' | tr '/' '_')
    filename=$(basename "$file")

    # Copy the file to the target directory with the new name
    cp "$file" "$target_directory/${subdir}_${filename}"
done

echo "All .log files have been copied to $target_directory."

