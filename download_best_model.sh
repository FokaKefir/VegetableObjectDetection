#!/bin/bash

# Download the trained vegetable detector model from Dropbox
# Requires: curl

OUTPUT_DIR="${1:-.}"
OUTPUT_FILE="${OUTPUT_DIR}/best_model.pth"
URL="https://www.dropbox.com/scl/fi/xeqdf23h0h7oetw426vrw/best_model.pth?rlkey=o7h2bfsew8wslv3bfl2dams55&st=hqnifqah&dl=1"

mkdir -p "${OUTPUT_DIR}"

echo "Downloading best model checkpoint..."
if curl -L --fail -o "${OUTPUT_FILE}" "${URL}"; then
    echo "Download complete: ${OUTPUT_FILE}"
else
    echo "Download failed"
    exit 1
fi