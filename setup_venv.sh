#!/bin/bash
# Setup Python venv for zing-world-model.
#
# Deviates from the original script in two ways, both for this specific
# GB300 box (aarch64, CUDA 13.2), matching fixes already applied tonight to
# MIND/H3-World's setup scripts on the same machine:
#   1. No hardcoded python3.11 + unprompted `sudo apt install` fallback --
#      uses whatever python3 is already on PATH instead.
#   2. torch/torchvision/torchaudio from cu132, unpinned (not the original's
#      torch==2.9.1 from cu128) -- a pinned version can silently not exist
#      on a different CUDA index and pip falls back to something wrong
#      instead of erroring (confirmed the hard way on H3-World tonight).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "=========================================="
echo "Zing World Model - Setup"
echo "=========================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found on PATH." >&2
    exit 1
fi

echo "Using $(python3 --version 2>&1)"
echo ""

# Remove old venv if it exists
if [ -d ".venv" ]; then
    echo "Removing old venv..."
    rm -rf .venv
fi

# Create venv
echo "[1/4] Creating Python venv..."
python3 -m venv .venv
source .venv/bin/activate

echo ""
echo "[2/4] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo ""
echo "[3/4] Installing PyTorch (cu132, unpinned -- see script header)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132

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
