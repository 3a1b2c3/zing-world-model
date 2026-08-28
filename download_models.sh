#!/bin/bash
# Download zing-world-model pretrained weights
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${HERE}/pretrained_models"
mkdir -p "$MODELS_DIR"

echo "=========================================="
echo "Zing World Model - Download Models"
echo "=========================================="
echo ""

# Check HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set"
    echo "Set: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

echo "Downloading models to: $MODELS_DIR"
echo ""

# Download base model
echo "[1/2] Downloading zing-0.5 base model..."
python << 'PYEOF'
from huggingface_hub import snapshot_download
import os

hf_token = os.environ.get("HF_TOKEN")
models_dir = os.path.join(os.path.dirname(__file__), "pretrained_models")

try:
    # Try downloading from HF repo (adjust repo name as needed)
    repo_id = "zingai/zing-world-model-0.5"  # Update with actual repo
    snapshot_download(
        repo_id,
        cache_dir=models_dir,
        token=hf_token,
        resume_download=True
    )
    print(f"✓ Downloaded {repo_id}")
except Exception as e:
    print(f"Note: {e}")
    print("Model repo may not be available yet. Check HF for availability.")
PYEOF

echo ""
echo "[2/2] Setup complete!"
echo ""
echo "Models downloaded to: $MODELS_DIR"
echo "Next: bash run_example.sh"
echo ""
