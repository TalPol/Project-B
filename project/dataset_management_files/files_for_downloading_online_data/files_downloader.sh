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

# Check if the destination folder exists, create if not
if [ ! -d "$DESTINATION" ]; then
    mkdir -p "$DESTINATION"
    echo "Created folder: $DESTINATION"
else
    # If the folder already exists, create a versioned folder
    VERSION=1
    while [ -d "$DESTINATION" ]; do
        VERSION=$((VERSION + 1))
        DESTINATION="${DESTINATION}_ver${VERSION}"
    done
    mkdir -p "$DESTINATION"
    echo "Created folder: $DESTINATION"
fi

# Use curl to fetch the webpage and grep to find links
curl -s "$URL" | grep -oP 'href="\K[^"]*\.(('"$FILE_TYPE"'|tar\.gz))' | while read -r link; do
    # Download the file
    file_name=$(basename "$link")
    curl -o "$DESTINATION/$file_name" "$link"
    echo "Downloaded and logged: $file_name"
done

echo "All files matching the criteria have been downloaded and logged."
