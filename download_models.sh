#!/bin/bash
# Download zing-0.5 pretrained weights.
#
# A wrapper around _download_models_helper.py, which holds the actual logic --
# shared with download_models.bat so the two platforms cannot drift apart. The
# repository previously carried three separate copies of the same
# snapshot_download call, each with different bugs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${MODELS_DIR:-$HERE/pretrained_models}"

echo "=========================================="
echo "Zing-0.5 model download"
echo "=========================================="
echo

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set." >&2
  echo "  export HF_TOKEN=<token>" >&2
  exit 1
fi

# The venv's interpreter when it exists, so huggingface_hub is found without
# the caller having to activate anything first.
PYTHON="$HERE/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3 || command -v python)"
fi

"$PYTHON" "$HERE/_download_models_helper.py" "$MODELS_DIR"

echo
echo "=========================================="
echo "Next: bash run_rtx5090.sh"
echo "=========================================="
