# Quick Start: Run Examples

## Windows (via WSL2)

The `.bat` files automate the entire setup and run process in WSL2.

### RTX 5090 (32GB)

```batch
run_rtx5090.bat
```

- Checks for WSL2
- Syncs project to WSL2
- Sets up Python venv
- Installs dependencies
- Runs inference with `33/5` config
- Syncs outputs back to Windows

**Expected time:** ~5-10 minutes (first run includes setup)

### RTX 6000 (48GB)

```batch
run_rtx6000.bat
```

Same process, but uses the faster `97/9` config.

**Expected time:** ~2-4 minutes (first run includes setup)

## Subsequent Runs

After the first run, setup is cached in WSL2. Runs 2+ are faster:

```batch
run_rtx5090.bat     # ~5-10 min inference only
run_rtx6000.bat     # ~2-4 min inference only
```

## Manual Setup (WSL2)

If you prefer to manage WSL2 directly:

```bash
# In WSL2 terminal
cd ~/zing-world-model
source .venv/bin/activate

# RTX 5090
bash run_rtx5090.sh

# RTX 6000
bash run_rtx6000.sh
```

See [WSL2_SETUP.md](WSL2_SETUP.md) for detailed instructions.

## Requirements

- Windows 11 with WSL2 enabled
- NVIDIA GPU (RTX 5090 or RTX 6000)
- CUDA toolkit installed in WSL2:
  ```bash
  wsl sudo apt install nvidia-cuda-toolkit
  ```

## Troubleshooting

**WSL2 not found:**
```powershell
wsl --install
```

**CUDA not found in WSL2:**
```bash
wsl sudo apt install nvidia-cuda-toolkit
```

**Permission denied:**
- Run PowerShell/Command Prompt as Administrator

**Outputs not syncing:**
- Check `%USERPROFILE%\zing-world-model\outputs` in WSL2
- Or access via: `\\wsl.localhost\Ubuntu\home\%USERNAME%\zing-world-model\outputs`

## Output Files

After each run, video outputs are saved to:
```
outputs/case3_rtx5090/
outputs/case3_rtx6000/
```

Each contains `.mp4` files (video frames rendered).
