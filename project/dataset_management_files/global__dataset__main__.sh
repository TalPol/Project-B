#!/bin/bash

#Arguments:
DOWNLOAD_FOLDER=$1

# Proper process of each file type: for all the first step is the downloader

# zipped up fasta files: unzipper, squ_handler, sofa5, fapo5

# zipped up fast5 files: unzipper, sofa5, stm_converter, fapo5

# regular fasta files: squ handler 
#
#
#
#
#
#
#
#
#
#
#Websites names and links assignnig:
MONASH_UNI1="$https://bridges.monash.edu/articles/dataset/Reference_genomes/7676135?file=14260238"

MONASH_UNI2="$https://bridges.monash.edu/articles/dataset/Raw_fast5s/7676174?file=14260541"

#MONASH_UNI:
mkdir -p "$DOWNLOAD_FOLDER/monash_uni"
mkdir -p "$DOWNLOAD_FOLDER/monash_uni/zip"
mkdir -p "$DOWNLOAD_FOLDER/monash_uni/fasta"
mkdir -p "$DOWNLOAD_FOLDER/monash_uni/fast5"
mkdir -p "$DOWNLOAD_FOLDER/monash_uni/slow5"
mkdir -p "$DOWNLOAD_FOLDER/monash_uni/pod5"

mkdir -p "$DOWNLOAD_FOLDER/monash_uni/fast5/single_read"
mkdir -p "$DOWNLOAD_FOLDER/monash_uni/fast5/multi_read"

./files_for_downloading_online_data/files_downloader.sh "$MONASH_UNI1" "${fasta.gz}" "$DOWNLOAD_FOLDER" "${monash_uni}"
./files_for_downloading_online_data/data_processing_files/unzip.sh "$DOWNLOAD_FOLDER" "${fasta.gz}" "monash_uni" "${.fasta}"

./files_for_downloading_online_data/data_processing_files/squ_handler.sh "$DOWNLOAD_FOLDER/monash_uni"
# One output is a slow5, converted later into a fast5 multi-read
slow5tools s2f "$DOWNLOAD_FOLDER/monash_uni/slow5" -d fast5

./files_for_downloading_online_data/files_downloader.sh "$MONASH_UNI2" "${.tar.gz}" "$DOWNLOAD_FOLDER/monash_uni/zip"
./files_for_downloading_online_data/data_processing_files/unzip.sh "$DOWNLOAD_FOLDER/monash_uni/zip" "$DOWNLOAD_FOLDER/monash_uni/fast5/single_read" "${.tar.gz}"

./dataset_management_files/slow5_to_fast_5_to_pod5/fast5_single_to_multi_converter.sh "$DOWNLOAD_FOLDER/monash_uni/fast5"
./dataset_management_files/slow5_to_fast_5_to_pod5/fapo5_manager.sh "$DOWNLOAD_FOLDER/monash_uni/fast5/multi_read" "$DOWNLOAD_FOLDER/monash_uni/pod5"



