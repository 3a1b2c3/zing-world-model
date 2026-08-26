from __future__ import annotations

import json
import re
from pathlib import Path


def iter_message_jsonl(path: str | Path):
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as source:
        for line_index, line in enumerate(source):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                sample = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source_path} line {line_index + 1} is not valid JSON") from exc
            if not isinstance(sample, dict):
                raise ValueError(f"{source_path} line {line_index + 1} must be a JSON object")
            yield line_index, sample


def output_stem(sample: dict, line_index: int) -> str:
    raw = str(sample.get("sample_id") or f"sample_{line_index:06d}")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return stem or f"sample_{line_index:06d}"


def write_media(output_dir: str | Path, stem: str, frames, fps: int) -> Path:
    import numpy as np
    import torch

    tensor = torch.as_tensor(frames, dtype=torch.uint8).cpu()
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if tensor.shape[0] == 1:
        from PIL import Image

        target = target_dir / f"{stem}.png"
        Image.fromarray(tensor[0].numpy()).save(target)
        return target

    target = target_dir / f"{stem}.mp4"
    try:
        from torchvision.io import write_video

        write_video(str(target), tensor, fps=int(fps))
        return target
    except ImportError:
        pass

    import av

    array = tensor.numpy()
    with av.open(str(target), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width = int(array.shape[2])
        stream.height = int(array.shape[1])
        stream.pix_fmt = "yuv420p"
        for image in array:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return target
