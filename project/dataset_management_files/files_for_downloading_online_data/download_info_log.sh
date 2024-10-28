#!/bin/bash

# Arguments
WEBSITE_URL=$1
FILE_TYPE=$2
DOWNLOAD_DEST=$3
FILE_NAME=$4
LOG_FILE_PATH=$5

# Check if all arguments are provided
if [ -z "$WEBSITE_URL" ] || [ -z "$FILE_TYPE" ] || [ -z "$DOWNLOAD_DEST" ] || [ -z "$FILE_NAME" ] || [ -z "$LOG_FILE_PATH" ]; then
    echo "Usage: $0 <URL> <FILE_TYPE> <DESTINATION> <FILE_NAME> <LOG_FILE_PATH>"
    exit 1
fi


# Determine the current number of downloads in the log file
if [ -f "$LOG_FILE_PATH" ]; then
    download_number=$(($(grep -c "^download" "$LOG_FILE_PATH") + 1))
else
    download_number=1
fi

# Append a blank line if the log file already contains data
if [ -s "$LOG_FILE_PATH" ]; then
    echo "" >> "$LOG_FILE_PATH"
fi

# Append download information to the log file
echo "download $download_number: \"$WEBSITE_URL\" \"$FILE_TYPE\" \"$DOWNLOAD_DEST\" \"$FILE_NAME\"" >> "$LOG_FILE_PATH"
