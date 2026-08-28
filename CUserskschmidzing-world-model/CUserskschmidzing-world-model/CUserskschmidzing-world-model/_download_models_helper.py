#!/usr/bin/env python3
"""Helper script to download zing-world-model pretrained weights."""

import os
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: _download_models_helper.py <models_dir>")
        sys.exit(1)

    models_dir = sys.argv[1]
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    try:
        repo_id = "seedleap/zing-0.5"
        print(f"Downloading {repo_id}...")
        snapshot_download(
            repo_id,
            cache_dir=models_dir,
            token=hf_token,
            resume_download=True
        )
        print(f"✓ Downloaded {repo_id}")
    except Exception as e:
        print(f"ERROR: {e}")
        print("Model repo may not be available yet. Check HF for availability.")
        sys.exit(1)

if __name__ == "__main__":
    main()
