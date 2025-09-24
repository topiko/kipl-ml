#!/bin/bash

# Loop over all directories with numeric names
for dir in [0-9]*; do
  if [ -d "$dir" ]; then
    # Zero-pad the directory number to 4 digits
    dirnum=$(printf "%04d" "$dir")

    # Loop over all .log files in the directory
    for file in "$dir"/*.log; do
      # Extract the original filename (e.g., 1.log)
      base=$(basename "$file")
      # Rename to include the directory prefix
      mv "$file" "$dir/${dirnum}-$base"
    done
  fi
done
