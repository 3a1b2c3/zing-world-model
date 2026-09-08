#!/usr/bin/env python3
"""Download zing-0.5 pretrained weights.

The single implementation behind download_models.sh and download_models.bat,
which are thin wrappers. Previously the same snapshot_download call was
duplicated across three scripts with different bugs in each; the shell copy
resolved its output directory from ``__file__`` inside a heredoc, where that
name does not exist, then swallowed the resulting NameError and exited zero --
so a failed download reported success.

    python _download_models_helper.py <models_dir>
"""

import os
import sys
from pathlib import Path

REPO_ID = "seedleap/zing-0.5"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: _download_models_helper.py <models_dir>")

    models_dir = Path(sys.argv[1]).resolve()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN is not set. export HF_TOKEN=<token>")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("ERROR: huggingface_hub is not installed. Run setup_venv.sh first.")

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"repo:  {REPO_ID}")
    print(f"into:  {models_dir}")
    # Reported because a partial download looks like a complete one on disk:
    # the count only settles once the transfer finishes.
    print(f"start: {sum(1 for _ in models_dir.rglob('*') if _.is_file())} files already present")
    print()

    try:
        path = snapshot_download(REPO_ID, cache_dir=str(models_dir), token=token)
    except Exception as error:
        # Deliberately fatal. The previous shell version caught this, printed a
        # note about the repo maybe not existing yet, and exited zero, which
        # made a missing model look like a successful setup.
        sys.exit(f"ERROR: download failed: {type(error).__name__}: {error}")

    files = [p for p in Path(path).rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"\ndone: {len(files)} files, {total / 1024**3:.2f} GB")
    print(f"      {path}")


if __name__ == "__main__":
    main()
