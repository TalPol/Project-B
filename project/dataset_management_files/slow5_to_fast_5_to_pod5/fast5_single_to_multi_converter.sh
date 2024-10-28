#!/bin/bash

#Arguments
INPUT_FOLDER_PATH=$1
OUTPUT_FOLDER_PATH=$2
BATCH_SIZE=$3

if [ -z "$INPUT_FOLDER_PATH" ] || [ -z "$OUTPUT_FOLDER_PATH" ]; then
    echo "Usage: $0 <INPUT_MOTHER_FOLDER_PATH> <OUTPUT_MOTHER_FOLDER_PATH>"
    exit 1
fi



