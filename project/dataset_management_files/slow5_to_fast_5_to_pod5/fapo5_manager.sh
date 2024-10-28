#!/bin/bash

#Arguments
INPUT_MOTHER_FOLDER_PATH=$1
OUTPUT_MOTHER_FOLDER_PATH=$2

# Check if both arguments are provided
if [ -z "$INPUT_MOTHER_FOLDER_PATH" ] || [ -z "$OUTPUT_MOTHER_FOLDER_PATH" ]; then
    echo "Usage: $0 <INPUT_MOTHER_FOLDER_PATH> <OUTPUT_MOTHER_FOLDER_PATH>"
    exit 1
fi

# Iterate over each subfolder in the input mother folder
for input_folder in "$INPUT_MOTHER_FOLDER_PATH"/*/; do
    # Get the basename of the current folder (e.g., "folder_kobra_kai_fast5s" or "folder1")
    folder_name=$(basename "$input_folder")
    # Determine the appropriate output folder name
    if [[ "$folder_name" == *"fast5"* ]]; then
        # Replace "fast5" with "pod5" if it exists
        output_folder_name="${folder_name//fast5/pod5}"
    else
        # Add "_pod5" to the end if "fast5" is not in the name
        output_folder_name="${folder_name}_pod5"
    fi

    if [ ! -d "$OUTPUT_MOTHER_FOLDER_PATH/pod5/$output_folder_name" ]; then
        # Create the output folder in the mother folder of outputs
        mkdir -p "$OUTPUT_MOTHER_FOLDER_PATH/pod5/$output_folder_name"
        echo "Created folder: $OUTPUT_MOTHER_FOLDER_PATH/pod5/$output_folder_name"
    fi
    
    if [ -n "$(find "$OUTPUT_MOTHER_FOLDER_PATH/pod5/$output_folder_name" -type f)" ]; then
        #Print the error massage
        echo "Folder: $OUTPUT_MOTHER_FOLDER_PATH/pod5/$output_folder_name contains files prior to the cuurent user's request regarding it."
    else
        # Call the Python script with two variables
        #python3 /app/slow5_to_fast_5_to_pod5/fapo5.py "$INPUT_MOTHER_FOLDER_PATH/$input_folder" "$OUTPUT_MOTHER_FOLDER_PATH/$output_folder_name"
        pod5 convert fast5 "$input_folder"/*.fast5 --output $OUTPUT_MOTHER_FOLDER_PATH/pod5/$output_folder_name --one-to-one "$input_folder"
    fi

done





