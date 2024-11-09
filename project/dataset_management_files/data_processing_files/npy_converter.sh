#!/bin/bash

# args
INPUT_PATH=$1
OUTPUT_PATH=$2

<<comment
for pod5_file in $INPUT_PATH/pod5/*; do

    file_name=$(basename "$pod5_file")
    folder_name="${file_name//.pod5/}"
    mkdir -p "$OUTPUT_PATH/"${folder_name}""
    echo "$INPUT_PATH/fasta/${folder_name}_ref.fasta"
    python ./preprocess.py "$INPUT_PATH/fasta/${folder_name}_ref.fasta" $pod5_file "$OUTPUT_PATH/"${folder_name}""
done
comment

for slow5_file in $INPUT_PATH/slow5/*; do

    file_name=$(basename "$slow5_file")
    folder_name="${file_name//.slow5/}"
    mkdir -p "$OUTPUT_PATH/"${folder_name}""
    echo "$INPUT_PATH/sam/${folder_name}.sam"
    python dataset_management_files/preprocess_11_04.py $slow5_file "$INPUT_PATH/sam/${folder_name}.sam"  "$OUTPUT_PATH/"${folder_name}""
done