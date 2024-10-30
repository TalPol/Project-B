#!/bin/bash

#Arguments:
FOLDER_PATH=$1

N=3

for file in $FOLDER_PATH/fasta/*; do
    if [[ "$file" != *_sim_* ]]; then
        file_name=$(basename "$file")
        #folder_name="${file_name//fasta/}"
        folder_name="${file_name//_reference.fasta/}"
        echo $folder_name
        output_file_name="${file_name//.fasta/}"
        output_file_name="${output_file_name//fasta/}"

        mkdir -p "$FOLDER_PATH/slow5/friendly/"${folder_name}"_sim_slow5"
        mkdir -p "$FOLDER_PATH/slow5/aggressive/"${folder_name}"_sim_slow5"

        for ((index=1; index<=N; index++)); do
            #Example of code line in squ_handler:
            #/app/squigulator-v0.4.0/squigulator
            #/app/squigulator/fasta/Haemophilus_haemolyticus_M1C132_1_reference.fasta
            #-x dna-r9-min
            #-o /app/squigulator/sim_slow5/Haemophilus_haemolyticus_M1C132_1.slow5 
            #-n 8669 -c /app/misc/Haemophilus.sam 
            #-a /app/misc/Haemophilus.paf 
            #-q /app/squigulator/sim_fasta/Haemophilus_haemolyticus_M1C132_1.fasta  
            
            # ONT friendly:    
            ./squigulator-v0.4.0/squigulator \
            $file \
            -x dna-r9-min \
            -o "$FOLDER_PATH/slow5/friendly/"${folder_name}"_sim_slow5"/${folder_name}_sim_${index}_friendly.slow5 \
            -n 4000 \
            -q $FOLDER_PATH/fasta/${folder_name}_sim_${index}_ref_friendly.fasta \
            -c $FOLDER_PATH/sam/${folder_name}_sam_sim_${index}_friendly.sam \
            --seed $index --ont-friendly=yes
            
            #ONT aggressive
            ./squigulator-v0.4.0/squigulator \
            $file \
            -x dna-r9-min \
            -o "$FOLDER_PATH/slow5/aggressive/"${folder_name}"_sim_slow5"/${folder_name}_sim_${index}_aggressive.slow5 \
            -n 4000 \
            -q $FOLDER_PATH/fasta/${folder_name}_sim_${index}_ref_aggressive.fasta \
            -c $FOLDER_PATH/sam/${folder_name}_sam_sim_${index}_aggressive.sam \
            --seed $index --ont-friendly=no
            


        done
        #iteration N
    fi
done
