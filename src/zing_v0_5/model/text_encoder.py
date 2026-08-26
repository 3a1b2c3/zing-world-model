from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, UMT5EncoderModel


class WanTextEncoder(torch.nn.Module):
    def __init__(self, pretrained_dir: str | Path, max_length: int):
        super().__init__()
        root = Path(pretrained_dir)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            root / "text_encoder", dtype=torch.bfloat16
        ).encoder.eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
        self.max_length = max_length

    def encode(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids, mask = output.input_ids, output.attention_mask
        if ids.shape[1] < 512:
            padding = 512 - ids.shape[1]
            ids = F.pad(ids, (0, padding), value=self.tokenizer.pad_token_id)
            mask = F.pad(mask, (0, padding))
        device = next(self.text_encoder.parameters()).device
        ids, mask = ids.to(device), mask.to(device)
        lengths = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        for embedding, length in zip(context, lengths.tolist()):
            embedding[length:] = 0
        prompt_lengths = lengths.clamp_min(512).to(torch.int32)
        flattened = torch.cat(
            [embedding[:length] for embedding, length in zip(context, prompt_lengths.tolist())], dim=0
        )
        return flattened, prompt_lengths
