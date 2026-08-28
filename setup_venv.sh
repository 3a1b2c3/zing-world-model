#!/bin/bash
# Setup Python venv for zing-world-model (Python 3.11 required)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "=========================================="
echo "Zing World Model - Setup"
echo "=========================================="
echo ""

# Check for Python 3.11, install if missing
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 not found, installing..."
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv
    if ! command -v python3.11 &> /dev/null; then
        echo "ERROR: Failed to install Python 3.11"
        exit 1
    fi
fi

echo "Using Python 3.11"
python3.11 --version
echo ""

# Remove old venv if it exists
if [ -d ".venv" ]; then
    echo "Removing old venv..."
    rm -rf .venv
fi

# Create venv with Python 3.11
echo "[1/4] Creating Python venv..."
python3.11 -m venv .venv
source .venv/bin/activate

echo ""
echo "[2/4] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo ""
echo "[3/4] Installing PyTorch (cu121 + torchaudio)..."
pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "[4/4] Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Source venv: source .venv/bin/activate"
echo "2. Download models: bash download_models.sh"
echo "3. Run example: bash run_rtx5090.sh (or run_rtx6000.sh)"
echo ""
