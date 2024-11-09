#!/bin/bash

# IMPORTANT NOTE:
# This file is relevant only after you have downloaded the data you wish to use and extracted the compressed files,
# and it is recommended to use it only after the data was properly divided to the subets of train, test and validation.

# correct input form: ./dataset_management_files/29_10_files_proc/dataset_handler.sh ./squigulator/squ_handler_test



#Arguments:
FOLDER_PATH=$1
OUTPUT_PATH=$2

./dataset_management_files/data_processing_files/squ_handler.sh "$FOLDER_PATH"
./dataset_management_files/data_processing_files/npy_converter.sh "$FOLDER_PATH" "$OUTPUT_PATH"
rm -r $FOLDER_PATH/slow5/
rm -r $FOLDER_PATH/sam/
python dataset_management_files/dataset.py "$OUTPUT_PATH"
#./slow5tools-v1.2.0/slow5tools s2f "$FOLDER_PATH/slow5/" -d "$FOLDER_PATH/fast5"
#Cannot remove slow5 files
#rm -r $FOLDER_PATH/slow5/
#./dataset_management_files/slow5_to_fast_5_to_pod5/fapo5_manager.sh "$FOLDER_PATH/fast5" "$FOLDER_PATH/pod5"
#rm -r $FOLDER_PATH/fast5/



