#!/bin/bash

# Work assupmtiopns:
# 1) All the files that we download were compressed, current assumption is either "*.gz* or "*.tar.gz"
# 2) 

#Arguments:
DOWNLOAD_FOLDER=$1

#Websites names and links assignnig:
MONASH_UNI1="https://bridges.monash.edu/articles/dataset/Reference_genomes/7676135?file=14260238"

MONASH_UNI2="https://bridges.monash.edu/articles/dataset/Raw_fast5s/7676174?file=14260541"

#MONASH_UNI:
mkdir -p "$DOWNLOAD_FOLDER/zip"
mkdir -p "$DOWNLOAD_FOLDER/fasta"
mkdir -p "$DOWNLOAD_FOLDER/fast5"
mkdir -p "$DOWNLOAD_FOLDER/slow5"
mkdir -p "$DOWNLOAD_FOLDER/pod5"

mkdir -p "$DOWNLOAD_FOLDER/fast5/single_read"
mkdir -p "$DOWNLOAD_FOLDER/fast5/multi_read"

./files_for_downloading_online_data/files_downloader.sh "$MONASH_UNI1" "fasta" "$DOWNLOAD_FOLDER"
./files_for_downloading_online_data/data_processing_files/unzip.sh "$DOWNLOAD_FOLDER/zip" "fasta"

./files_for_downloading_online_data/data_processing_files/squ_handler.sh "$DOWNLOAD_FOLDER"
# The slow5 files are multi reads
slow5tools s2f "$DOWNLOAD_FOLDER/slow5" -d ./$DOWNLOAD_FOLDER/fast5/multi_read

./files_for_downloading_online_data/files_downloader.sh "$MONASH_UNI2" "tar.gz" "$DOWNLOAD_FOLDER"
./files_for_downloading_online_data/data_processing_files/unzip.sh "$DOWNLOAD_FOLDER/zip" "fast5"

./dataset_management_files/slow5_to_fast_5_to_pod5/fast5_single_to_multi_converter.sh "$DOWNLOAD_FOLDER/fast5"
./dataset_management_files/slow5_to_fast_5_to_pod5/fapo5_manager.sh "$DOWNLOAD_FOLDER/fast5/multi_read" "$DOWNLOAD_FOLDER/pod5"



