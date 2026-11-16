"""Sinusoidal flow-time features + the 2-layer MLP that projects them.

Adapted from huggingface/diffusers ``models/embeddings.py`` (Copyright 2024 The HuggingFace Team, Apache-2.0),
via the Interactive World Simulator (MIT); modified. See THIRD_PARTY.md."""
from __future__ import annotations

import math

import torch
from torch import nn


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """DDPM-style sinusoidal features for a 1-D ``(N,)`` or 2-D ``(N, M)`` tensor of (fractional) indices."""
    if timesteps.ndim not in (1, 2):
        raise ValueError("timesteps must be a 1D or 2D tensor")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        0, half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[..., None].float() * emb
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[..., half_dim:], emb[..., :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Module wrapper around :func:`get_timestep_embedding`."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class TimestepEmbedding(nn.Module):
    """Linear -> SiLU -> Linear projection of the sinusoidal features (no hidden normalization)."""

    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: int | None = None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim or time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = sample.to(self.linear_1.weight.dtype)
        return self.linear_2(self.act(self.linear_1(sample)))
