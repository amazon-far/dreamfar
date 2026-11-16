"""Whole-episode batches from precomputed latent shards (for rollouts and the demo)."""
from __future__ import annotations

import json

import torch
from torch.utils.data import default_collate

from dreamfar.data.latent_dataset import LatentShardDataset, load_catalog


def list_episodes(latent_root: str, heldout_path: str | None = None) -> tuple[list, dict]:
    """(episodes sorted by asset_id, shard meta). With `heldout_path` only the held-out ids are kept."""
    episodes, meta = load_catalog(latent_root)
    if heldout_path:
        held = set(json.load(open(heldout_path))["asset_ids"])
        episodes = [e for e in episodes if e.asset_id in held]
    return sorted(episodes, key=lambda e: e.asset_id), meta


def load_episode(model, ep, meta: dict, device: str, cap: int = 300, contact_dir: str | None = None,
                 frame_skip: int = 3) -> dict:
    """One episode as a B=1 batch at 10 Hz: latent (1,T,V,C,h,w), raw EE action (1,T,20), raw joint
    state (1,T,14), frame_idx (1,T) and, with `contact_dir`, per-arm contact (1,T,2).
    T = min((size-1)//frame_skip + 1, cap)."""
    T = min((ep.size - 1) // frame_skip + 1, cap)
    ds = LatentShardDataset([ep], tuple(meta["latent_shape"]), T, contact_dir, latent_dtype=meta["latent_dtype"],
                            frame_skip=frame_skip, random_start_jitter=False,
                            contact_min_dwell=model.contact_min_dwell, fps=float(meta["fps"]))
    batch = default_collate([ds[0]])
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
