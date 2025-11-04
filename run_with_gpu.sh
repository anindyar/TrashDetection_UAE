#!/bin/bash
# Wrapper script to run trash detection with GPU support
# Uses venv_gpu which has access to system PyTorch with AMD GPU (ROCm) support

echo "========================================"
echo "Running with AMD GPU Support"
echo "========================================"
echo ""

# Check GPU availability
./venv_gpu/bin/python -c "
import torch
print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'GPU Available: {torch.cuda.is_available()}')
print('')
"

# Run the detection script with GPU-enabled venv
./venv_gpu/bin/python enhanced_trash_detect_v2.py "$@"
