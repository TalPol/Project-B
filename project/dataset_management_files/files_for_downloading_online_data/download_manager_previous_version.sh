#!/bin/bash

# Arguments: URL, file type, memory limit (GB), base path, folder name
URL=$1
FILE_TYPE=$2
MEMORY_LIMIT_GB=$3
BASE_PATH=$4
FOLDER_NAME=$5

# Create the folder with version handling
TARGET_FOLDER="${BASE_PATH}/${FOLDER_NAME}"


if [ ! -d TARGET_FOLDER ]; then
	mkdir -p "$TARGET_FOLDER"
	echo "Created folder: $TARGET_FOLDER"
else
	CREATED=0
	VERSION=2
	while [ "$CREATED" -eq 0 ]; do
		TARGET_FOLDER="${BASE_PATH}/${FOLDER_NAME}_ver${VERSION}"
		if [ ! -d "$TARGET_FOLDER" ]; then
			mkdir -p "$TARGET_FOLDER"
			echo "Created folder: $TARGET_FOLDER"
			CREATED=1
		fi
		((VERSION++))
	done
fi

# Create the download log file in the new folder
touch "${TARGET_FOLDER}/download_log_${FOLDER_NAME}.txt"

# Call the Python script with arguments
python3 /app/squigulator/files_for_downloading_online_data/files_downloader.py "$URL" "$FILE_TYPE" "$MEMORY_LIMIT_GB" "$TARGET_FOLDER"

# Check if Python script executed successfully
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed or memory limit reached."
fi