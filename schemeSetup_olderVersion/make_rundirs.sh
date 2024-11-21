#!/bin/bash
# Check if the command line argument is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num> <workpath>"
    exit 1
fi

num=$1
workpath=$2

cd $workpath

source_dir="R$num"
cp -r J$num  $source_dir
# Check if the source directory exists
if [ ! -d "$source_dir" ]; then
    echo "Source directory $source_dir not found."
    exit 1
fi

# Use find to locate all .gjf files in the source directory
find "$source_dir" -type f -name "*.gjf" -print0 | while IFS= read -r -d '' gjf_file; do
    # Extract the filename without extension
    filename=$(basename "$gjf_file" .gjf)
    
    # Create a directory with the filename (without extension)
    target_dir="$source_dir/$filename"
    mkdir -p "$target_dir"
    
    # Move the .gjf file to the new directory with ".com" extension
    mv "$gjf_file" "$target_dir/$filename.com"
    
    #echo "Moved $gjf_file to $target_dir/$filename.com"
done
echo "Made running directories"

