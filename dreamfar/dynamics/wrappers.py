"""``EinopsWrapper``: run a (B,C,T,H,W) module on a frame-major (T,B,C,H,W) stream.

The dynamics net is registered as ``self.module``, so checkpoint keys are ``dynamics.module.*``.
"""
from __future__ import annotations

import torch
from einops import parse_shape, rearrange
from torch import nn


class EinopsWrapper(nn.Module):
    """Reshape the input via einops, run ``module``, reshape the grid output back.

    When the module returns a tuple, ONLY ``out[0]`` (the grid) is rearranged back; the extra
    outputs (state velocity, contact logits, MCP list) are already in the caller's layout."""

    def __init__(self, from_shape: str, to_shape: str, module: nn.Module):
        super().__init__()
        self.module = module
        self.from_shape = from_shape
        self.to_shape = to_shape

    def forward(self, x: torch.Tensor, *args, **kwargs):
        axes_lengths = parse_shape(x, pattern=self.from_shape)
        x = rearrange(x, f"{self.from_shape} -> {self.to_shape}")
        out = self.module(x, *args, **kwargs)
        if isinstance(out, tuple):
            grid = rearrange(out[0], f"{self.to_shape} -> {self.from_shape}", **axes_lengths)
            return (grid,) + tuple(out[1:])
        return rearrange(out, f"{self.to_shape} -> {self.from_shape}", **axes_lengths)
