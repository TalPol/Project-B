#!/bin/bash

#Arguments:
INPUT=$1
ZIPPING_TYPE=$2
SOURCE_NAME=$3
OUT_TYPE=$4

for file in "$INPUT/$SOURCE_NAME/zip"; do
    if [[ "$file"==*".${ZIPPING_TYPE}" ]]; then
        file_name=$(basename "$file")
        output_path="$OUTPUT//${file_name%."$OUT_TYPE"}"
        mkdir -p "$output_path"
        unzip -q "$file" -d "$output_path"
    fi
done
