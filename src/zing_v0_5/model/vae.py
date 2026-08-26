from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderKLWan


class WanVAE(torch.nn.Module):
    def __init__(self, pretrained_dir: str | Path):
        super().__init__()
        self.model = AutoencoderKLWan.from_pretrained(Path(pretrained_dir) / "vae")
        self.model.eval().requires_grad_(False)
        self.model.clear_cache()
        self.mean = torch.tensor(self.model.config.latents_mean, dtype=torch.float32)
        self.std = torch.tensor(self.model.config.latents_std, dtype=torch.float32)

    def _stats(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean.to(device=reference.device, dtype=reference.dtype).view(1, -1, 1, 1, 1)
        std = self.std.to(device=reference.device, dtype=reference.dtype).view(1, -1, 1, 1, 1)
        return mean, std

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype != torch.uint8:
            raise ValueError("reference frames must use uint8 pixels")
        dtype = next(self.model.parameters()).dtype
        pixels = frames.to(device=next(self.model.parameters()).device, dtype=dtype).div(127.5).sub(1.0)
        mean, std = self._stats(pixels)
        self.model.clear_cache()
        latent = (self.model.encode(pixels).latent_dist.mode() - mean) / std
        self.model.clear_cache()
        return latent.float().permute(0, 2, 1, 3, 4)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        dtype = next(self.model.parameters()).dtype
        latents = latents.to(device=next(self.model.parameters()).device, dtype=dtype)
        mean, std = self._stats(latents)
        value = latents.permute(0, 2, 1, 3, 4) * std + mean
        self.model.clear_cache()
        decoded = self.model.decode(value).sample.float()
        self.model.clear_cache()
        return decoded.permute(0, 2, 1, 3, 4)
