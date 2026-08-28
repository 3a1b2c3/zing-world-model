# Zing-0.5 on WSL2 Setup Guide

Zing-0.5 requires Linux, so you must run it in WSL2 on Windows. This guide sets up both RTX 5090 and RTX 6000 example runs.

## Prerequisites

- WSL2 with Ubuntu 22.04 LTS (or newer)
- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit in WSL2 (install via `apt`)

## WSL2 CUDA Setup (One-time)

In WSL2 terminal:

```bash
# Update package lists
sudo apt update

# Install CUDA toolkit (this installs cuDNN and related libraries)
sudo apt install -y nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version
nvidia-smi
```

## Copy Project to WSL2

From PowerShell on Windows:

```powershell
# Copy project to WSL2 home (one-time)
wsl cp -r /mnt/c/workspace/world/zing-world-model ~/zing-world-model

# Or create a symlink in WSL2 (persistent):
wsl ln -s /mnt/c/workspace/world/zing-world-model ~/zing-world-model
```

## Setup Environment in WSL2

In WSL2 terminal:

```bash
cd ~/zing-world-model

# Create venv with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CUDA 12.1)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## Download Models (Windows or WSL2)

**On Windows:**
```batch
cd C:\workspace\world\zing-world-model
set HF_TOKEN=your_huggingface_token
download_models.bat
```

Models are cached in `pretrained_models/` and will be accessible from WSL2 via `/mnt/c/workspace/world/zing-world-model/pretrained_models`.

## Run Examples

### RTX 5090 (32GB) — Uses `33/5` config

In WSL2:
```bash
cd ~/zing-world-model
source .venv/bin/activate
bash run_rtx5090.sh
```

Or run manually:
```bash
python -m zing_v0_5 \
  --pretrained-dir ./pretrained_models \
  --checkpoint ./pretrained_models/models--seedleap--zing-0.5 \
  --messages examples/case3_action_t2v.jsonl \
  --output-dir outputs/case3_rtx5090 \
  --local-attn-size 33 \
  --sink-size 5 \
  --seed 0
```

**Expected time:** ~5-10 minutes for case3 (31 frames)

### RTX 6000 (48GB) — Uses `97/9` config (default)

In WSL2:
```bash
cd ~/zing-world-model
source .venv/bin/activate
bash run_rtx6000.sh
```

Or run manually:
```bash
python -m zing_v0_5 \
  --pretrained-dir ./pretrained_models \
  --checkpoint ./pretrained_models/models--seedleap--zing-0.5 \
  --messages examples/case3_action_t2v.jsonl \
  --output-dir outputs/case3_rtx6000 \
  --local-attn-size 97 \
  --sink-size 9 \
  --seed 0
```

**Expected time:** ~2-4 minutes for case3 (31 frames)

## Other Examples

Available examples (same JSON format):

| Example | Frames | Config |
|---------|--------|--------|
| `case3_action_t2v.jsonl` | 31 | Action control + text init |
| `case4_action_ti2v.jsonl` | 25 | Action control + image init |
| `case6_prompt_switch_t2v.jsonl` | 30 | Prompt switching |
| `case7_action_prompt_switch_t2v.jsonl` | 45 | Both action + prompt |

Replace `--messages examples/case3_action_t2v.jsonl` with any of the above.

## Troubleshooting

**"CUDA out of memory"**
- RTX 5090: Already using `33/5` config (lightest). Try shorter outputs.
- RTX 6000: Switch to `33/5` config with `--local-attn-size 33 --sink-size 5`

**"Module not found" errors**
- Ensure venv is activated: `source .venv/bin/activate`
- Re-install: `pip install -r requirements.txt`

**Slow performance**
- Check GPU load: `nvidia-smi` (should be ~95%+)
- Ensure `CUDA_VISIBLE_DEVICES=0` is set (or correct GPU number)
- RTX 6000's `97/9` config is faster; RTX 5090's `33/5` is slower but fits memory

## Memory Breakdown

**RTX 5090 (32GB) with `33/5`:**
- Model: ~18GB
- Cache + activations: ~14GB
- Total: ~32GB ✓

**RTX 6000 (48GB) with `97/9`:**
- Model: ~18GB
- Cache + activations: ~30GB
- Total: ~48GB ✓

**RTX 6000 with `33/5`** (conservative):
- Same model, but smaller cache = faster mem access
- ~22GB total (lots of headroom)
