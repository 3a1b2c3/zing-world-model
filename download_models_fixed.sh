#!/bin/bash
# Download zing-0.5 models from HuggingFace

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$HERE/pretrained_models"

echo "=========================================="
echo "Zing-0.5 Model Download"
echo "=========================================="
echo ""

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set"
    echo "Run: export HF_TOKEN=your_token_here"
    exit 1
fi

mkdir -p "$MODELS_DIR"

echo "Downloading to: $MODELS_DIR"
echo "Token: ${HF_TOKEN:0:10}..."
echo ""

python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import os
import sys

models_dir = os.path.expandvars("$MODELS_DIR")
token = os.environ.get("HF_TOKEN")

if not token:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

try:
    print("Downloading seedleap/zing-0.5...")
    snapshot_download(
        "seedleap/zing-0.5",
        cache_dir=models_dir,
        token=token
    )
    print("✓ Download complete!")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
PYEOF

echo ""
echo "Models ready in: $MODELS_DIR"
echo "Next: bash run_rtx5090.sh"
echo ""
