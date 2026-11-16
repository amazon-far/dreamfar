"""Rollout seeding regimes, built as kwargs for ``DynamicsModule.rollout``.

* ``cold``: GT frame 0 only (the rest of the committed window is mask-padded).
* ``dup``:  duplicated cold start: the committed window is pre-filled with copies of GT frame 0, so
            the first prediction sees a full (static) context. Action/state/frame_idx are padded by
            the same ``pad`` so buffer index j holds episode frame max(0, j - pad).
* ``mid``:  seed the first ``seed_sec`` seconds of GT frames (mid-episode take-over).
"""
from __future__ import annotations

from typing import Optional

import torch


def committed_window(model) -> int:
    """Committed slots in the streaming layout: n_tokens - dyn_flow_steps."""
    return int(model.n_tokens) - int(model.dyn_flow_steps)


def _prepend(t: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return t
    head = t[:, :1]
    reps = [1, pad] + [1] * (t.dim() - 2)
    return torch.cat([head.repeat(*reps), t], dim=1)


def build_regime(regime: str, model, z_gt: torch.Tensor, action: torch.Tensor, state: torch.Tensor,
                 frame_idx: torch.Tensor, contact: Optional[torch.Tensor] = None,
                 n_dup: Optional[int] = None, seed_sec: float = 15.0, fps: int = 10) -> dict:
    """Returns {z_hist, action, state, frame_idx, seed_latents, seed_contact, pad, n_seed, regime}."""
    T = z_gt.shape[1]
    if regime == "cold":
        return {"z_hist": z_gt[:, :1], "action": action, "state": state, "frame_idx": frame_idx,
                "seed_latents": None, "seed_contact": None, "pad": 0, "n_seed": 1, "regime": regime}
    if regime == "mid":
        n_seed = max(1, min(T - 1, int(round(seed_sec * fps))))
        return {"z_hist": z_gt[:, :1], "action": action, "state": state, "frame_idx": frame_idx,
                "seed_latents": z_gt[:, :n_seed],
                "seed_contact": (contact[0, :n_seed] if contact is not None else None),
                "pad": 0, "n_seed": n_seed, "regime": regime}
    if regime == "dup":
        n_dup = committed_window(model) if n_dup is None else int(n_dup)
        n_dup = max(2, min(n_dup, T))
        pad = n_dup - 1
        z_seed = z_gt[:, :1].repeat(1, n_dup, 1, 1, 1)
        sc = contact[0, :1].repeat(n_dup, 1) if contact is not None else None
        return {"z_hist": z_gt[:, :1], "action": _prepend(action, pad), "state": _prepend(state, pad),
                "frame_idx": _prepend(frame_idx, pad), "seed_latents": z_seed, "seed_contact": sc,
                "pad": pad, "n_seed": n_dup, "regime": regime}
    raise ValueError(f"regime must be cold|dup|mid, got {regime!r}")
