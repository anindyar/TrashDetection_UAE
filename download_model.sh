#!/bin/bash
# Helper script to download the Roboflow model using the virtual environment

# Check if API key is provided
if [ -z "$1" ]; then
    echo "Usage: ./download_model.sh YOUR_ROBOFLOW_API_KEY"
    echo ""
    echo "Get your API key from: https://app.roboflow.com/settings/api"
    exit 1
fi

# Run the download script using the venv Python
./venv/bin/python download_roboflow_model.py "$1"
