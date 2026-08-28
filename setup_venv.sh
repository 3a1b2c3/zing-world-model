#!/bin/bash
# Setup Python venv for zing-world-model
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "=========================================="
echo "Zing World Model - Setup"
echo "=========================================="
echo ""

# Create venv
echo "[1/3] Creating Python venv..."
python3 -m venv .venv
source .venv/bin/activate

echo ""
echo "[2/3] Installing dependencies..."
pip install --upgrade pip setuptools wheel

# Install torch with CUDA 13.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132

# Install other dependencies
pip install -r requirements.txt

echo ""
echo "[3/3] Setup complete!"
echo ""
echo "Next steps:"
echo "1. Source venv: source .venv/bin/activate"
echo "2. Download models: bash download_models.sh"
echo "3. Run example: bash run_example.sh"
echo ""
