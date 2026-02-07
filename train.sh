#!/bin/bash

# Exit on error
set -e

echo "Preparing environment for training..."

# Ensure Kaggle directory exists
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "WARNING: ~/.kaggle/kaggle.json not found."
    echo "Please download your Kaggle API token from https://www.kaggle.com/settings"
    echo "and place it at ~/.kaggle/kaggle.json"
fi

# Run training
echo "Starting training (Weight Optimization)..."
python3 adaptive_fwi.py
