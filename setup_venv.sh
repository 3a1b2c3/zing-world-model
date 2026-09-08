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
# FLASH_ATTENTION_FORCE_BUILD=TRUE is what makes this a source build at all.
# Without it flash-attn's setup.py downloads a prebuilt wheel from its GitHub
# releases whenever one matches the Python/CUDA/platform triple. Those wheels
# are linked against whichever torch they were built with, so on a venv with a
# different torch the install *succeeds in seconds* and then fails at import:
#
#   ImportError: flash_attn_2_cuda...so: undefined symbol:
#   _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
#
# which is c10::cuda::c10_cuda_check_implementation -- a torch symbol whose
# signature changed between versions. An instantaneous install here is the
# tell; a real build takes 20-40 minutes.
#
# --no-cache-dir for the same reason one level up: pip will otherwise reuse a
# wheel it cached from an earlier run against a different torch.
#
# Unpinned (was 2.6.3, then 2.8.3). Both were recorded as "failed to build"
# with no compiler error ever captured -- plausibly this same silent wheel
# mismatch rather than a compile failure, since 2.8.3.post1 does build here.
#
# MAX_JOBS caps compile parallelism; the default can exhaust memory on this
# box. Raise it if there is headroom.
#
# Not allowed to abort the script. It is the last step, everything else has
# already succeeded, and `set -e` would otherwise throw that away over an
# optional accelerator. The build log is kept because the two previous
# failures here were never diagnosed -- the compiler error scrolled past and
# the script died before anything could be read.
FLASH_LOG="$HERE/flash-attn-build.log"
if pip install flash-attn --no-build-isolation > "$FLASH_LOG" 2>&1; then
  echo "      flash-attn installed."
  rm -f "$FLASH_LOG"
else
  echo ""
  echo "      WARNING: flash-attn failed to build. Everything else is installed."
  echo "      Full log: $FLASH_LOG"
  echo ""
  echo "      Last error lines:"
  grep -iE "error|Error [0-9]|fatal" "$FLASH_LOG" | tail -15 | sed "s/^/        /"
  echo ""
  echo "      Note: src/zing_v0_5/model/attention.py calls flash_attn_varlen_func"
  echo "      with no fallback, so inference will fail on import until this is"
  echo "      resolved. FlashAttention 2 targets sm80-sm90; this box is Blackwell,"
  echo "      so an unsupported-architecture error here is expected rather than a"
  echo "      misconfiguration."
fi

echo ""
echo "=========================================="
if [ -f "$FLASH_LOG" ]; then
  echo "Setup complete, except flash-attn (see warning above)."
else
  echo "Setup complete!"
fi
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Source venv: source .venv/bin/activate"
echo "2. Download models: bash download_models.sh"
echo "3. Run example: bash run_rtx5090.sh (or run_rtx6000.sh)"
echo ""
