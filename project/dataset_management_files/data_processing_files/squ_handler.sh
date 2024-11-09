#!/bin/bash

#Arguments:
FOLDER_PATH=$1

N_AMP=2
N_SEED=2

mkdir -p "$FOLDER_PATH/slow5"
mkdir -p "$FOLDER_PATH/sam"
for file in $FOLDER_PATH/fasta/*; do
    if [[ "$file" != *_sim_* ]]; then
        file_name=$(basename "$file")
        #folder_name="${file_name//fasta/}"
        folder_name="${file_name//_reference.fasta/}"
        folder_name="${folder_name//.fasta/}"
        echo $folder_name
        output_file_name="${file_name//.fasta/}"
        output_file_name="${output_file_name//fasta/}"

        #mkdir -p "$FOLDER_PATH/slow5/"${folder_name}"_sim_slow5"
        # loop goes over amp index and seed 
        for ((amp_index=1; amp_index<=N_AMP; amp_index++)); do
            for ((seed_index=1; seed_index<=N_SEED; seed_index++)); do   
                index=$(( (amp_index-1) * 10 + seed_index )) 
                ./squigulator-v0.4.0/squigulator \
                $file \
                -x dna-r10-min \
                -o "$FOLDER_PATH/slow5/"${folder_name}_sim_$index.slow5 \
                -n 5000 -r 10000 --dwell-std 4.0 --dwell-mean 13.0 \
                -a $FOLDER_PATH/sam/${folder_name}_sim_$index.sam \
                --amp-noise $amp_index \
                --seed $index --ont-friendly=yes
            
            # -q $FOLDER_PATH/fasta/${folder_name}_sim_${index}_ref.fasta \
            done    
        done
        
    fi
done
