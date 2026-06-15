#!/bin/bash

# Download vegetable dataset from Kaggle
# Requires: curl

OUTPUT_DIR="${1:-.}"
OUTPUT_FILE="${OUTPUT_DIR}/dataset.zip"

echo "Downloading vegetable dataset from Kaggle..."
curl -L -o "${OUTPUT_FILE}" \
  "https://www.kaggle.com/api/v1/datasets/download/ayyuce/vegetables"

if [ $? -eq 0 ]; then
    echo "Download complete: ${OUTPUT_FILE}"
else
    echo "Download failed"
    exit 1
fi
