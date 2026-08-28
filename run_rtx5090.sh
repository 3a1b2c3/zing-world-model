#!/bin/bash
# Zing-0.5 on RTX 5090 (32GB VRAM)
# Uses lighter sliding-window config (33/5)

set -e
cd "$(dirname "$0")"

PRETRAINED_DIR="$(pwd)/pretrained_models"
OUTPUT_DIR="$(pwd)/outputs"

# Find the snapshot directory in HuggingFace cache
find_snapshot_dir() {
    local search_dir="$1"

    # Check if pretrained/ exists directly (already extracted at root)
    if [ -d "$search_dir/pretrained/text_encoder" ]; then
        echo "$search_dir"
        return 0
    fi

    # Search in HuggingFace cache: models--seedleap--zing-0.5/snapshots/<hash>/pretrained/
    if [ -d "$search_dir/models--seedleap--zing-0.5" ]; then
        local latest=$(find "$search_dir/models--seedleap--zing-0.5/snapshots" -maxdepth 1 -type d | sort -r | head -1)
        if [ -d "$latest/pretrained/text_encoder" ]; then
            echo "$latest"
            return 0
        fi
    fi

    return 1
}

if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "ERROR: Models directory not found at $PRETRAINED_DIR"
    exit 1
fi

SNAPSHOT_DIR=$(find_snapshot_dir "$PRETRAINED_DIR") || {
    echo "ERROR: Could not find Zing-0.5 models"
    echo "Expected: pretrained/text_encoder/, pretrained/tokenizer/, pretrained/vae/, generator/model.pt"
    echo ""
    echo "Current contents:"
    ls -la "$PRETRAINED_DIR" 2>/dev/null || true
    exit 1
}

PRETRAINED_PATH="$SNAPSHOT_DIR/pretrained"
CHECKPOINT_PATH="$SNAPSHOT_DIR/generator/model.pt"

if [ ! -d "$PRETRAINED_PATH/text_encoder" ]; then
    echo "ERROR: text_encoder/ not found at $PRETRAINED_PATH"
    exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: generator/model.pt not found at $CHECKPOINT_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Zing-0.5 on RTX 5090 (32GB)"
echo "=========================================="
echo "Pretrained: $PRETRAINED_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
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
  --pretrained-dir "$PRETRAINED_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --messages "examples/case3_action_t2v.jsonl" \
  --output-dir "$OUTPUT_DIR/case3_rtx5090" \
  --local-attn-size 33 \
  --sink-size 5 \
  --seed 0

echo ""
echo "Done! Output saved to: $OUTPUT_DIR/case3_rtx5090"
echo ""
