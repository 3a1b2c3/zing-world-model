from __future__ import annotations

import inspect
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional. flash-attn ships prebuilt wheels linked against a specific torch
# ABI, so on a venv with a different torch it imports with
# "undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation..." -- and no
# release on PyPI is currently built against torch 2.14. Absent it, the varlen
# path below falls back to padded SDPA, which is slower on ragged batches but
# has no build step and tracks whatever torch is installed.
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


os.environ.setdefault("TRITON_MAX_BLOCK_X", "8192")
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch._inductor.config.realize_opcount_threshold = 100
torch._dynamo.config.recompile_limit = 1024


class CompiledSegment:
    compiled = {}
    mode = None

    @classmethod
    def get(cls, function, enabled: bool):
        if not enabled:
            return function
        if function not in cls.compiled:
            options = {}
            if "recompile_limit" in inspect.signature(torch.compile).parameters:
                options["recompile_limit"] = 1024
            cls.compiled[function] = torch.compile(function, dynamic=True, mode=cls.mode, **options)
        return cls.compiled[function]


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, compile_fusion: bool):
        super().__init__()
        self.eps = eps
        self.compile_fusion = compile_fusion
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return CompiledSegment.get(self._norm, self.compile_fusion and value.is_cuda)(value, self.weight, self.eps)

    @staticmethod
    def _norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        value_float = value.float()
        return (value_float * torch.rsqrt(value_float.pow(2).mean(dim=-1, keepdim=True) + eps)).type_as(value) * weight


def make_rope_freqs(dim: int, num_heads: int, maximum: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def table(length: int, width: int) -> torch.Tensor:
        positions = torch.arange(length)
        frequencies = 1.0 / torch.pow(10000, torch.arange(0, width, 2).to(torch.float64).div(width))
        angles = torch.outer(positions, frequencies)
        return torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).float()

    head_dim = dim // num_heads
    temporal_width = head_dim - 4 * (head_dim // 6)
    spatial_width = 2 * (head_dim // 6)
    return table(maximum, temporal_width), table(maximum, spatial_width), table(maximum, spatial_width)


def compute_rope(
    positions: torch.Tensor, temporal: torch.Tensor, height: torch.Tensor, width: torch.Tensor
) -> torch.Tensor:
    maxima = positions.max(dim=0).values
    if int(maxima[0]) >= temporal.shape[0] or int(maxima[1]) >= height.shape[0] or int(maxima[2]) >= width.shape[0]:
        raise ValueError("RoPE position exceeds generator.rope_max_seq_len")
    return torch.cat((temporal[positions[:, 0]], height[positions[:, 1]], width[positions[:, 2]]), dim=1)


def apply_rope(value: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    sequence = value.shape[-3]
    head_dim = value.shape[-1]
    shaped = rope.reshape(*([1] * (value.dim() - 3)), sequence, 1, head_dim // 2, 2)
    cosine, sine = shaped[..., 0], shaped[..., 1]
    real, imaginary = value[..., 0::2].float(), value[..., 1::2].float()
    rotated = torch.stack((real * cosine - imaginary * sine, real * sine + imaginary * cosine), dim=-1)
    return rotated.flatten(-2).to(value.dtype)


def _sdpa_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
) -> torch.Tensor:
    """Variable-length attention via padded SDPA, for when flash-attn is absent.

    Inputs and output are packed the way flash_attn_varlen_func expects:
    (total_tokens, heads, head_dim), with per-sequence lengths given separately.
    Sequences are padded to the batch maximum, attended with a mask that keeps
    padding out of the softmax, then repacked.

    Memory cost is the padding: a batch whose lengths vary widely allocates
    batch * max_len rather than the sum of lengths. That is the whole reason the
    varlen kernel exists, so this is a fallback rather than an equivalent.
    """
    batch = query_lengths.numel()
    max_query = int(query_lengths.max().item())
    max_key = int(key_lengths.max().item())
    heads, head_dim = query.shape[-2], query.shape[-1]

    padded_query = query.new_zeros((batch, max_query, heads, head_dim))
    padded_key = key.new_zeros((batch, max_key, heads, head_dim))
    padded_value = value.new_zeros((batch, max_key, heads, head_dim))
    # Built as a boolean keep-mask over (batch, 1, query, key); the head axis
    # broadcasts. False entries are excluded from the softmax entirely, which
    # is what stops padded keys contributing.
    mask = torch.zeros((batch, 1, max_query, max_key), dtype=torch.bool, device=query.device)

    query_offset = 0
    key_offset = 0
    for index in range(batch):
        query_length = int(query_lengths[index].item())
        key_length = int(key_lengths[index].item())
        padded_query[index, :query_length] = query[query_offset:query_offset + query_length]
        padded_key[index, :key_length] = key[key_offset:key_offset + key_length]
        padded_value[index, :key_length] = value[key_offset:key_offset + key_length]
        mask[index, 0, :query_length, :key_length] = True
        query_offset += query_length
        key_offset += key_length

    # SDPA wants (batch, heads, sequence, head_dim).
    attended = F.scaled_dot_product_attention(
        padded_query.transpose(1, 2),
        padded_key.transpose(1, 2),
        padded_value.transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2)

    packed = query.new_empty(query.shape)
    query_offset = 0
    for index in range(batch):
        query_length = int(query_lengths[index].item())
        packed[query_offset:query_offset + query_length] = attended[index, :query_length]
        query_offset += query_length
    return packed


def flash_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    deterministic: bool,
) -> torch.Tensor:
    if flash_attn_varlen_func is None:
        return _sdpa_varlen(query, key, value, query_lengths, key_lengths)

    cumulative_query = F.pad(query_lengths.to(torch.int32).cumsum(0), (1, 0)).to(torch.int32)
    cumulative_key = F.pad(key_lengths.to(torch.int32).cumsum(0), (1, 0)).to(torch.int32)
    return flash_attn_varlen_func(
        query,
        key,
        value,
        cumulative_query,
        cumulative_key,
        int(query_lengths.max().item()),
        int(key_lengths.max().item()),
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        deterministic=deterministic,
    )


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, qk_norm: bool, compile_fusion: bool, deterministic: bool):
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dimension must be divisible by the head count")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.deterministic = deterministic
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        history: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden.shape
        query = self.q(hidden)
        key = self.k(hidden)
        if isinstance(self.norm_q, WanRMSNorm):
            query = WanRMSNorm._norm(query, self.norm_q.weight, self.norm_q.eps)
            key = WanRMSNorm._norm(key, self.norm_k.weight, self.norm_k.eps)
        query = query.reshape(batch, sequence, self.num_heads, self.head_dim)
        key = key.reshape(batch, sequence, self.num_heads, self.head_dim)
        value = self.v(hidden).reshape(batch, sequence, self.num_heads, self.head_dim)
        history_key, history_value = history
        raw_key = key if history_key is None else torch.cat((history_key, key), dim=1)
        full_value = value if history_value is None else torch.cat((history_value, value), dim=1)
        query = apply_rope(query, query_rope)
        rotated_key = apply_rope(raw_key, key_rope)
        key_sequence = rotated_key.shape[1]
        query_lengths = torch.full((batch,), sequence, device=hidden.device, dtype=torch.int32)
        key_lengths = torch.full((batch,), key_sequence, device=hidden.device, dtype=torch.int32)
        attended = flash_attention_varlen(
            query.reshape(batch * sequence, self.num_heads, self.head_dim),
            rotated_key.reshape(batch * key_sequence, self.num_heads, self.head_dim),
            full_value.reshape(batch * key_sequence, self.num_heads, self.head_dim),
            query_lengths,
            key_lengths,
            self.deterministic,
        )
        attended = attended.reshape(batch, sequence, self.num_heads * self.head_dim)
        return self.o(attended), key, value


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, qk_norm: bool, compile_fusion: bool, deterministic: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.deterministic = deterministic
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        cached: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, sequence, _ = hidden.shape
        query = self.norm_q(self.q(hidden)).reshape(batch * sequence, self.num_heads, self.head_dim)
        key, value = cached
        created = None
        if key is None:
            key = self.norm_k(self.k(context)).reshape(context.shape[0], self.num_heads, self.head_dim)
            value = self.v(context).reshape(context.shape[0], self.num_heads, self.head_dim)
            created = (key, value)
        query_lengths = torch.full((batch,), sequence, device=hidden.device, dtype=torch.int32)
        attended = flash_attention_varlen(
            query, key, value, query_lengths, context_lengths.to(torch.int32), self.deterministic
        )
        return self.o(attended.reshape(batch, sequence, -1)), created
