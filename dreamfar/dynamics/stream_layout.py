"""Sequence-layout helpers for the streaming layout ``[sink | memory | context c]``.

Frame axis, length ``N = n_sink + n_mem + n_ctx``: the SINK (learned token bank, never noised, no
RoPE, no cond), MEMORY (own encoder, attends only to itself, rotary position ``-squash(dt)``), and the
CONTEXT ``c`` (causal, rotary positions ``[0 .. n_ctx-1]``). Mask polarity follows SDPA: True = attend.
"""
from __future__ import annotations

from typing import Optional

import torch


def dt_to_rope_pos(dt: torch.Tensor, dt_span: float = 64.0, dt_ref: float = 1000.0) -> torch.Tensor:
    """Memory Δt (>=0 frames back from ``c[0]``) -> bounded NEGATIVE rotary position.

    ``pos = -dt_span * log1p(dt) / log1p(dt_ref)``, clamped to ``[-dt_span, 0]``; Δt=0 -> 0."""
    dt = dt.clamp(min=0.0)
    pos = -dt_span * torch.log1p(dt) / float(torch.log1p(torch.tensor(float(dt_ref))))
    return pos.clamp(min=-dt_span, max=0.0)


def build_stream_rope_pos(
    n_mem: int,
    n_ctx: int,
    mem_dt: torch.Tensor,
    dt_span: float = 64.0,
    dt_ref: float = 1000.0,
    n_sink: int = 1,
) -> torch.Tensor:
    """Per-frame temporal rotary positions for ``[sink | memory | c]``: sink 0, memory
    ``dt_to_rope_pos(Δt)``, c ``[0 .. n_ctx-1]``. mem_dt ``(n_mem,)`` -> ``(N,)``; ``(B,n_mem)`` -> ``(B,N)``."""
    batched = mem_dt.dim() == 2
    dev, dt_ = mem_dt.device, mem_dt.dtype if mem_dt.is_floating_point() else torch.float32
    mem_pos = dt_to_rope_pos(mem_dt.to(dt_), dt_span=dt_span, dt_ref=dt_ref)
    assert mem_pos.shape[-1] == n_mem, f"mem_dt last dim {mem_pos.shape[-1]} != n_mem {n_mem}"
    ctx_pos = torch.arange(n_ctx, device=dev, dtype=dt_)
    sink_pos = torch.zeros(n_sink, device=dev, dtype=dt_)
    if not batched:
        return torch.cat([sink_pos, mem_pos, ctx_pos], dim=0)                  # (N,)
    B = mem_pos.shape[0]
    return torch.cat([sink_pos.expand(B, n_sink), mem_pos, ctx_pos.expand(B, n_ctx)], dim=1)  # (B,N)


def build_stream_attn_mask(
    n_mem: int,
    n_ctx: int,
    mem_valid: Optional[torch.Tensor] = None,
    ctx_valid: Optional[torch.Tensor] = None,
    device=None,
    n_sink: int = 1,
) -> torch.Tensor:
    """Temporal attention mask for ``[sink | memory | c]`` (True = attend).

    Rows: sink -> itself; memory -> itself; ``c[j]`` -> the sink, every VALID memory slot, and valid
    ``c[<=j]``. mem_valid: ``(n_mem,)``, ``(B,n_mem)`` or per-query-frame ``(B,n_ctx,n_mem)``;
    ctx_valid: ``(n_ctx,)`` or ``(B,n_ctx)``. Returns ``(N,N)`` when both are unbatched, else ``(B,N,N)``."""
    if mem_valid is None:
        mem_valid = torch.ones(n_mem, dtype=torch.bool, device=device)
    if ctx_valid is None:
        ctx_valid = torch.ones(n_ctx, dtype=torch.bool, device=device)
    mem_valid = mem_valid.to(torch.bool)
    ctx_valid = ctx_valid.to(torch.bool)
    per_frame = mem_valid.dim() == 3
    batched = per_frame or (mem_valid.dim() == 2) or (ctx_valid.dim() == 2)
    if per_frame:
        if ctx_valid.dim() == 1:
            ctx_valid = ctx_valid.unsqueeze(0).expand(mem_valid.shape[0], -1)
    elif batched:
        if mem_valid.dim() == 1:
            mem_valid = mem_valid.unsqueeze(0).expand(ctx_valid.shape[0], -1)
        if ctx_valid.dim() == 1:
            ctx_valid = ctx_valid.unsqueeze(0).expand(mem_valid.shape[0], -1)
    else:
        mem_valid = mem_valid.unsqueeze(0)
        ctx_valid = ctx_valid.unsqueeze(0)

    B = mem_valid.shape[0]
    dev = mem_valid.device
    N = n_sink + n_mem + n_ctx
    keep = torch.zeros((B, N, N), dtype=torch.bool, device=dev)

    # sink + memory rows: diagonal only.
    diag = torch.arange(n_sink + n_mem, device=dev)
    keep[:, diag, diag] = True

    c0 = n_sink + n_mem                                   # first `c` column/row
    keep[:, c0:, :n_sink] = True                          # c rows -> sink column(s)
    keep[:, c0:, n_sink:c0] = (mem_valid if per_frame     # c rows -> visible memory columns
                               else mem_valid[:, None, :].expand(B, n_ctx, n_mem))
    causal = torch.ones((n_ctx, n_ctx), dtype=torch.bool, device=dev).tril()
    keep[:, c0:, c0:] = causal[None] & ctx_valid[:, None, :].expand(B, n_ctx, n_ctx)
    return keep[0] if not batched else keep


def build_stream_cond_mask(n_mem: int, n_ctx: int, device=None, dtype=None,
                           n_sink: int = 1) -> torch.Tensor:
    """Action-cond FiLM gate over frames, ``(N,)`` in {0,1}: 0 on sink + memory, 1 on every ``c`` frame."""
    m = torch.zeros(n_sink + n_mem + n_ctx, device=device, dtype=dtype)
    m[n_sink + n_mem:] = 1.0
    return m
