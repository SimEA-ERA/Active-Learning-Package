#!/bin/bash

# Check if the command line argument is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num> <workpath>"
    exit 1
fi

num=$1
workpath=$2

cd $workpath

target_dir="L$num"

# Create the target directory if it doesn't exist
mkdir -p "$target_dir"

# Find all .log files in the current directory and copy them to the target directory
find R$num -maxdepth 2 -type f -name "*.log" -exec cp {} "$target_dir" \;

echo "Logs copied to $target_dir"

