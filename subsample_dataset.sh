#!/bin/bash

# Usage check
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <base_directory_path>"
    exit 1
fi

# Define the base directory where the original dataset is located
base_dir="$1"

# Check if the base directory exists
if [ ! -d "$base_dir" ]; then
    echo "Error: Base directory does not exist."
    exit 1
fi

# Extract the name of the dataset directory
DSDIRNAME=$(basename "$base_dir")  # Change 1: Extract directory name

# Initial settings
new_factor=2
initial_images=$(ls -1 "$base_dir/images/training" | wc -l)
echo "Initial number of images: $initial_images"

# Change to the parent directory of the base directory to store the subsampled datasets as siblings
cd "$(dirname "$base_dir")"  # Change 2: Move to the parent directory

# Initialize the first dataset as a direct copy of the original
first_dataset="${DSDIRNAME}_ds_1"  # Change 3: Use DSDIRNAME for naming
echo "Creating $first_dataset from $DSDIRNAME"
cp -r "$DSDIRNAME" "$first_dataset"

# Set the first_dataset as the last_dataset to start the loop
last_dataset="$first_dataset"

# Continue until fewer than 10 images would remain in the next round
while : ; do
    new_dataset="${DSDIRNAME}_ds_${new_factor}"  # Change 3: Use DSDIRNAME for naming
    echo "Creating $new_dataset from $last_dataset with subsampling factor $new_factor"

    # Create the new dataset directory
    mkdir -p "$new_dataset"
    if [ ! -d "$new_dataset" ]; then
        echo "Failed to create directory $new_dataset."
        exit 1
    fi

    # Copy the validation files
    mkdir -p "$new_dataset/images/validation"
    mkdir -p "$new_dataset/annotations/validation"
    cp -r "$last_dataset/images/validation" "$new_dataset/images/"
    cp -r "$last_dataset/annotations/validation" "$new_dataset/annotations/"

    # Copy only the required files
    mkdir -p "$new_dataset/images/training"
    mkdir -p "$new_dataset/annotations/training"
    find "$last_dataset/images/training/" -type f | sort | awk "NR % 2 == 0" | xargs -I {} cp {} "$new_dataset/images/training/"
    find "$last_dataset/annotations/training/" -type f | sort | awk "NR % 2 == 0" | xargs -I {} cp {} "$new_dataset/annotations/training/"

    # Update last_dataset to the new subsampled dataset directory
    last_dataset="$new_dataset"

    # Check if the next round would result in fewer than 10 images
    if [ $(($initial_images / $new_factor)) -lt 10 ]; then
        break
    fi
    # Prepare for the next iteration by updating new_factor
    next_factor=$(($new_factor * 2))
    new_factor=$next_factor
done

echo "Subsampling complete."
