#!/usr/bin/env python3
"""
Script to download YOLOv8 model from Roboflow

To use this script:
1. Get your API key from: https://app.roboflow.com/settings/api
2. Either:
   - Set the ROBOFLOW_API_KEY environment variable, OR
   - Pass your API key as a command-line argument

Usage:
    python download_roboflow_model.py YOUR_API_KEY
    # or
    export ROBOFLOW_API_KEY=your_key_here
    python download_roboflow_model.py
"""

from roboflow import Roboflow
import os
import sys

# Get API key from command-line argument or environment variable
if len(sys.argv) > 1:
    API_KEY = sys.argv[1]
else:
    API_KEY = os.environ.get("ROBOFLOW_API_KEY")

if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
    print("\nError: Please provide your Roboflow API key!")
    print("\nYou can get your API key from: https://app.roboflow.com/settings/api")
    print("\nUsage:")
    print("  python download_roboflow_model.py YOUR_API_KEY")
    print("  # or")
    print("  export ROBOFLOW_API_KEY=your_key_here")
    print("  python download_roboflow_model.py")
    sys.exit(1)

print("Initializing Roboflow...")
# Initialize Roboflow
rf = Roboflow(api_key=API_KEY)

# Project details from the URL
workspace = "drone-litter-detection"
project_name = "waste-detect-v-drone-1val9"

print(f"Accessing project: {workspace}/{project_name}")
# Get the project
project = rf.workspace(workspace).project(project_name)

print("Downloading dataset in YOLOv8 format...")
# Download the dataset in YOLOv8 PyTorch format
# version parameter: use the latest version or specify a version number
dataset = project.version(1).download("yolov8")

print(f"\n{'='*60}")
print(f"Dataset downloaded successfully!")
print(f"Location: {dataset.location}")
print(f"{'='*60}")
print(f"\nYou can now train or use this model with YOLOv8")
print(f"\nExample usage:")
print(f"  from ultralytics import YOLO")
print(f"  model = YOLO('yolov8n.pt')  # Load a pretrained model")
print(f"  model.train(data='{dataset.location}/data.yaml', epochs=100)")
