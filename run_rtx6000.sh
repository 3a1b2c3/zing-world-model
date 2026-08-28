#!/bin/bash
# Zing-0.5 on RTX 6000 (48GB VRAM)
# Uses default sliding-window config (97/9)

set -e
cd "$(dirname "$0")"

PRETRAINED_DIR="$(pwd)/pretrained_models"
CHECKPOINT="$PRETRAINED_DIR/models--seedleap--zing-0.5"
OUTPUT_DIR="$(pwd)/outputs"

if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "ERROR: Models not found at $PRETRAINED_DIR"
    echo "Run: download_models.bat (on Windows) or copy from Windows"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Zing-0.5 on RTX 6000 (48GB)"
echo "=========================================="
echo "Pretrained: $PRETRAINED_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo ""

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

echo "Running case3 (long action T2V)..."
python -m zing_v0_5 \
  --pretrained-dir "$PRETRAINED_DIR" \
  --checkpoint "$CHECKPOINT" \
  --messages "examples/case3_action_t2v.jsonl" \
  --output-dir "$OUTPUT_DIR/case3_rtx6000" \
  --local-attn-size 97 \
  --sink-size 9 \
  --seed 0

echo ""
echo "Done! Output saved to: $OUTPUT_DIR/case3_rtx6000"
echo ""
