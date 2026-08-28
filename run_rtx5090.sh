#!/bin/bash
# Zing-0.5 on RTX 5090 (32GB VRAM)
# Uses lighter sliding-window config (33/5)

set -e
cd "$(dirname "$0")"

PRETRAINED_DIR="$(pwd)/pretrained_models"
OUTPUT_DIR="$(pwd)/outputs"

if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "ERROR: pretrained_models/ not found"
    exit 1
fi

# Find the latest snapshot in HuggingFace cache
SNAPSHOT_DIR=$(find "$PRETRAINED_DIR/models--seedleap--zing-0.5/snapshots" -maxdepth 1 -type d -name "[a-f0-9]*" 2>/dev/null | sort -r | head -1)

if [ -z "$SNAPSHOT_DIR" ] || [ ! -d "$SNAPSHOT_DIR/pretrained/text_encoder" ]; then
    echo "ERROR: Could not find Zing-0.5 models"
    echo "Run: python -c \"from huggingface_hub import snapshot_download; snapshot_download('seedleap/zing-0.5', cache_dir='./pretrained_models', token='\$HF_TOKEN')\""
    exit 1
fi

PRETRAINED_PATH="$SNAPSHOT_DIR/pretrained"
CHECKPOINT_PATH="$SNAPSHOT_DIR/generator/model.pt"

mkdir -p "$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Zing-0.5 on RTX 5090 (32GB)"
echo "=========================================="
echo "Snapshot: $SNAPSHOT_DIR"
echo "Output: $OUTPUT_DIR/case3_rtx5090"
echo ""

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

echo "Running case3 (long action T2V)..."
python -m zing_v0_5 \
  --pretrained-dir "$PRETRAINED_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --messages "examples/case3_action_t2v.jsonl" \
  --output-dir "$OUTPUT_DIR/case3_rtx5090" \
  --local-attn-size 33 \
  --sink-size 5 \
  --seed 0

echo ""
echo "✓ Done! Output: $OUTPUT_DIR/case3_rtx5090"
echo ""
