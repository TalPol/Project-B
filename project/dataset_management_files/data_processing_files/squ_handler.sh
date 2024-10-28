#!/bin/bash

#Arguments:
FASTA_PATH=$1
SLOW5_PATH=$2

N=

for file in "$FASTA_PATH"; do
    if [ "$file" !=*"_sim_" ]; then
        file_name=$(basename "$file")
        ./fasta_squ.sh "$FASTA_PATH/$file" "$SLOW5_PATH" "$file_name"
    
    for ((index=1; index<N; index++)); do
        #Example of code line in squ_handler:
        #/app/squigulator-v0.4.0/squigulator
        #/app/squigulator/fasta/Haemophilus_haemolyticus_M1C132_1_reference.fasta
        #-x dna-r9-min
        #-o /app/squigulator/sim_slow5/Haemophilus_haemolyticus_M1C132_1.slow5 
        #-n 8669 -c /app/misc/Haemophilus.sam 
        #-a /app/misc/Haemophilus.paf 
        #-q /app/squigulator/sim_fasta/Haemophilus_haemolyticus_M1C132_1.fasta 
        ./dataset_management_files/data_processing_files/fasta_squ.sh "$FASTA_PATH" "$SLOW5_PATH" "$file" "$index"
    done
    fi
done