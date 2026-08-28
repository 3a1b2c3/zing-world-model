#!/bin/bash
# Zing-0.5 on RTX 6000 (48GB VRAM)
# Uses default sliding-window config (97/9)

set -e
cd "$(dirname "$0")"

PRETRAINED_DIR="$(pwd)/pretrained_models"
OUTPUT_DIR="$(pwd)/outputs"

# Find the actual checkpoint (huggingface_hub caches files in snapshots/<hash>/)
find_checkpoint() {
    local search_dir="$1"
    if [ -f "$search_dir/generator/model.pt" ]; then
        echo "$search_dir"
        return 0
    fi

    # Try huggingface cache structure
    for dir in "$search_dir"/models--seedleap--zing-0.5/snapshots/*/; do
        if [ -f "$dir/generator/model.pt" ]; then
            echo "$dir"
            return 0
        fi
    done

    return 1
}

if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "ERROR: Models directory not found at $PRETRAINED_DIR"
    echo "Download with: python -c \"from huggingface_hub import snapshot_download; snapshot_download('seedleap/zing-0.5', cache_dir='./pretrained_models', token='your_token')\""
    exit 1
fi

CHECKPOINT=$(find_checkpoint "$PRETRAINED_DIR") || {
    echo "ERROR: Could not find model.pt in $PRETRAINED_DIR"
    echo "Downloaded models should have structure: pretrained/... and generator/model.pt"
    ls -la "$PRETRAINED_DIR" 2>/dev/null || true
    exit 1
}

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Zing-0.5 on RTX 6000 (48GB)"
echo "=========================================="
echo "Pretrained: $CHECKPOINT"
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
  --pretrained-dir "$CHECKPOINT" \
  --checkpoint "$CHECKPOINT/generator/model.pt" \
  --messages "examples/case3_action_t2v.jsonl" \
  --output-dir "$OUTPUT_DIR/case3_rtx6000" \
  --local-attn-size 97 \
  --sink-size 9 \
  --seed 0

echo ""
echo "Done! Output saved to: $OUTPUT_DIR/case3_rtx6000"
echo ""
