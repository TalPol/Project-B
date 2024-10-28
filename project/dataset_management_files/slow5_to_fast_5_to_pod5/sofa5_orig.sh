#!/bin/bash

#Arguments:
INPUT_FOLDER_PATH=$1
OUTPUT_FOLDER_PATH=$2
INPUT_FILE_TYPE=$3
OUTPUT_FILE_TYPE=$4
#
# Create the missing folders with version handling


if [ ! -d TARGET_FOLDER_SLOW5 ]; then
	mkdir -p "$TARGET_FOLDER_SLOW5"
	echo "Created folder: $TARGET_FOLDER_SLOW5"
fi

if [ ! -d TARGET_FOLDER_FAST5 ]; then
	mkdir -p "$TARGET_FOLDER_FAST5"
	echo "Created folder: $TARGET_FOLDER_FAST5"
fi

if [ ! -d TARGET_FOLDER_POD5 ]; then
	mkdir -p "$TARGET_FOLDER_POD5"
	echo "Created folder: $TARGET_FOLDER_POD5"
fi

#Convert slow5 files to fast5 using slow5tools:

slow5tools slow5tofast5 "$TARGET_FOLDER_SLOW5" -d "$TARGET_FOLDER_FAST5"

#Call fapo5_manager with the needed info:

./fapo_manager.sh "$BASE_PATH"
