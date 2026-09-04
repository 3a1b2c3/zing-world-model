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
echo "[4/5] Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "[5/5] Installing flash-attn (--no-build-isolation, source build)..."
# flash-attn's setup.py imports torch at build time to detect CUDA
# arch/version -- pip's isolated build env doesn't include the venv's
# packages by default, so a plain `pip install flash-attn` fails with
# "ModuleNotFoundError: No module named 'torch'" even though torch is
# already installed above. --no-build-isolation fixes that. Building from
# source (no prebuilt wheel expected for this aarch64/cu132 combination),
# so this step is slow.
#
# Bumped from the project's original pin (2.6.3) to 2.8.3: 2.6.3's build
# failed on this box (compiler error, GB300/Blackwell/sm_100 -- exact cause
# not confirmed since the real error output wasn't captured, but the
# failure itself plus 2.6.3's age point at missing Blackwell support).
# 2.8.3 specifically chosen because it's independently confirmed elsewhere
# in this project's tooling to support Blackwell (sm_120, RTX 5090) --
# see MIND/.claude/skills/mind-benchmark: "flash_attn 2.8.3+cu128torch2.10
# -cp310" is the pinned, confirmed-working build for ViPE's own Blackwell
# setup. Not a guess -- real evidence, just from a different project.
pip install flash-attn==2.8.3 --no-build-isolation

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
