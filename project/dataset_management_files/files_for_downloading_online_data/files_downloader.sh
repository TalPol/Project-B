#!/bin/bash

# Arguments
URL=$1
FILE_TYPE=$2
DESTINATION=$3

# Check if all arguments are provided
if [ -z "$URL" ] || [ -z "$FILE_TYPE" ]; then
    echo "Usage: $0 <URL> <FILE_TYPE> <LOG_FILE_PATH>"
    exit 1
fi

curl -s "$URL" | grep -oP 'href="\K[^"]*'"$FILE_TYPE"'[^"]*\.(tar\.gz|gz)' | while read -r link; do
# Download the file
file_name=$(basename "$link")
curl -o "$DESTINATION/zip/$file_name" "$link"
echo "Downloaded and logged: $file_name"
done

echo "All files matching the criteria have been downloaded and logged."
