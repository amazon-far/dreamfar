"""Multi-Chunk Prediction (MCP) heads — Next Forcing (arXiv 2606.11187) on the streaming layout.

Training-only: a fuser over trunk features tapped at ``mcp_tap_layers``, then per depth k an
``in_proj`` + trunk-shaped block(s) that predict frame i+k (chained across depths). Discarded at inference.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torch.nn as nn


class McpHeads(nn.Module):
    """One fuser + per-depth (in_proj, blocks). The output head is the trunk's (none of its own)."""

    def __init__(self, dim: int, depths: int, blocks_per_depth: int, n_taps: int,
                 block_factory: Callable[[], nn.Module]):
        super().__init__()
        assert depths > 0 and blocks_per_depth > 0 and n_taps > 0
        self.depths = int(depths)
        self.blocks_per_depth = int(blocks_per_depth)
        self.fuser = nn.Sequential(nn.Linear(dim * n_taps, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.in_proj = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(self.depths)])
        self.blocks = nn.ModuleList([
            nn.ModuleList([block_factory() for _ in range(self.blocks_per_depth)])
            for _ in range(self.depths)
        ])
        for mod in (self.fuser, self.in_proj):
            for layer in mod.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def init_from_backbone(self, trunk_blocks: nn.ModuleList) -> bool:
        """Copy the LAST ``blocks_per_depth`` trunk blocks into every depth. False (fresh init kept)
        when a tail block is a different class (e.g. a joint full-attention block)."""
        src = list(trunk_blocks)[-self.blocks_per_depth:]
        for group in self.blocks:
            for dst, s in zip(group, src):
                if type(dst) is not type(s):
                    return False
        for group in self.blocks:
            for dst, s in zip(group, src):
                dst.load_state_dict(s.state_dict())
        return True

    def fuse(self, taps: List[torch.Tensor], ch_dim: int) -> torch.Tensor:
        """Fuse per-tap (B,C,...) features concatenated on the channel axis (dim 1)."""
        h = torch.cat(taps, dim=1).movedim(1, -1)
        return self.fuser(h).movedim(-1, 1)

    def project(self, k: int, prev: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """``in_proj_k(cat[h_prev, embed(x_{t_k})])`` on (B,C,...) tensors (channel at dim 1)."""
        h = torch.cat([prev, emb], dim=1).movedim(1, -1)
        return self.in_proj[k](h).movedim(-1, 1)


def mcp_tap_layers_for_depth(depth: int) -> List[int]:
    """The paper's tap fractions (1-indexed layers 4/12/20/30 of 30) rescaled to ``depth``, 0-indexed."""
    out = []
    for r in (4 / 30, 12 / 30, 20 / 30, 1.0):
        out.append(min(depth - 1, max(0, int(round(r * depth)) - 1)))
    return sorted(set(out))
