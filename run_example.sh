#!/bin/bash
# Run zing-world-model inference example
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Activate venv
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found"
    echo "Run: bash setup_venv.sh"
    exit 1
fi
source .venv/bin/activate

echo "=========================================="
echo "Zing World Model - Example"
echo "=========================================="
echo ""

# Set paths
PRETRAINED_DIR="${HERE}/pretrained_models"
CHECKPOINT="${PRETRAINED_DIR}/checkpoint.pt"  # Update with actual checkpoint name
MESSAGES="${HERE}/examples/sample_input.jsonl"
OUTPUT_DIR="${HERE}/outputs"

# Check paths
if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "ERROR: Models not found in $PRETRAINED_DIR"
    echo "Run: bash download_models.sh"
    exit 1
fi

if [ ! -f "$MESSAGES" ]; then
    echo "ERROR: Example input not found: $MESSAGES"
    echo "Create examples/sample_input.jsonl first"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Running inference..."
echo "  Pretrained: $PRETRAINED_DIR"
echo "  Checkpoint: $CHECKPOINT"
echo "  Input: $MESSAGES"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run inference
python -m zing_v0_5 \
  --pretrained-dir "$PRETRAINED_DIR" \
  --checkpoint "$CHECKPOINT" \
  --messages "$MESSAGES" \
  --output-dir "$OUTPUT_DIR" \
  --seed 42

echo ""
echo "✓ Complete!"
echo "Output: $OUTPUT_DIR"
echo ""
