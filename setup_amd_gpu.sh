#!/bin/bash
# Setup script to enable AMD GPU acceleration for trash detection system
# GPU: AMD Radeon 8050S/8060S (Strix Halo)

echo "=========================================="
echo "AMD GPU Acceleration Setup"
echo "=========================================="
echo ""

# Check current setup
echo "[1] Current PyTorch setup:"
./venv/bin/python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"
echo ""

# Option 1: Reinstall PyTorch with ROCm support
echo "[2] To enable AMD GPU acceleration, you need PyTorch with ROCm support"
echo ""
echo "Run these commands to install PyTorch ROCm 6.2:"
echo ""
echo "  # Uninstall current CUDA PyTorch"
echo "  ./venv/bin/pip uninstall torch torchvision torchaudio -y"
echo ""
echo "  # Install PyTorch with ROCm 6.2 support"
echo "  ./venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
echo ""
echo "  # Or use the system ROCm-enabled PyTorch:"
echo "  ./venv/bin/pip uninstall torch torchvision torchaudio -y"
echo "  # Then the system PyTorch (with ROCm) will be used"
echo ""

# Check if ROCm is available
echo "[3] Checking system ROCm:"
if command -v rocminfo &> /dev/null; then
    echo "  ROCm tools found!"
    rocminfo | grep -E "Marketing Name|Device Type" | head -5
else
    echo "  ROCm tools not in PATH (but system Python has ROCm support)"
fi
echo ""

echo "[4] GPU Information:"
lspci | grep -i 'vga\|display'
echo ""

echo "=========================================="
echo "Current Status: Running on CPU"
echo "After installing ROCm PyTorch, you'll get 3-5x speedup"
echo "=========================================="
