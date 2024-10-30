#!/bin/bash

#Arguments:
INPUT=$1
OUT_TYPE=$2

for file in "$INPUT/*"; do
    if [[ "$file" == *"*$OUT_TYPE*" ]]; then
        file_name=$(basename "$file")
        output_path="$INPUT/"$OUT_TYPE""
        mkdir -p "$output_path"
        unzip -q "$file" -d "$output_path"
    fi
done
