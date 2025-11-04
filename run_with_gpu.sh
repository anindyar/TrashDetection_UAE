#!/bin/bash
# Wrapper script to run trash detection with GPU support
# Uses system Python which has AMD GPU (ROCm) support

echo "========================================"
echo "Running with AMD GPU Support"
echo "========================================"
echo ""

# Check GPU availability
python3 -c "
import torch
print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'GPU Available: {torch.cuda.is_available()}')
print('')
"

# Run the detection script with system Python (has GPU support)
python3 enhanced_trash_detect_v2.py "$@"
