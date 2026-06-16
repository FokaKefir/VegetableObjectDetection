#!/bin/bash

# Download the custom vegetable dataset from Dropbox
# Requires: curl

OUTPUT_DIR="${1:-.}"
OUTPUT_FILE="${OUTPUT_DIR}/custom_dataset.zip"
URL="https://www.dropbox.com/scl/fi/ni48b5elv4l2esqma3euh/custom_dataset.zip?rlkey=u8blw19ywsfpc7jhee6y9yh3z&st=p419tqzd&dl=1"

mkdir -p "${OUTPUT_DIR}"

echo "Downloading custom vegetable dataset..."
if curl -L --fail -o "${OUTPUT_FILE}" "${URL}"; then
    echo "Download complete: ${OUTPUT_FILE}"
else
    echo "Download failed"
    exit 1
fi