#!/bin/bash

# Exit on error
set -e

# Default dataset (placeholder - user should change this)
DATASET_NAME=${1:-"your-kaggle-dataset-name"}

if [ "$DATASET_NAME" == "your-kaggle-dataset-name" ]; then
    echo "Usage: ./download_data.sh <dataset-name>"
    echo "Example: ./download_data.sh mczielinski/human-bold-signal-dataset"
    exit 1
fi

echo "Downloading dataset: $DATASET_NAME"

# Check for credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found. Cannot download from Kaggle."
    exit 1
fi

# Download
kaggle datasets download -d $DATASET_NAME --unzip -p ./data/

echo "Download complete. Data saved in ./data/"
