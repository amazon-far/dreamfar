"""Single-frame dataset over the release raw-episode format, for tokenizer training/validation.

    <raw_root>/index.json                          {"fps": 30, "episodes": [{"id", "task_name", "size"}, ...]}
    <raw_root>/<episode_id>/{top,left,right}.mp4   `size` time-aligned frames each (any resolution)

Pixel pipeline: decode (NHWC uint8) -> aspect-preserving letterbox to ``resolution`` x ``resolution`` ->
NCHW uint8 -> optional per-sample horizontal flip."""
from __future__ import annotations

import json
import os
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dreamfar.data.video import VideoReader, letterbox_resize as _letterbox_resize

VIEWS = ("top", "left", "right")


def load_index(raw_root: str) -> dict:
    with open(os.path.join(raw_root, "index.json")) as f:
        return json.load(f)


def split_episodes(index, heldout_path: Optional[str] = None, val_ratio: float = 0.01,
                   seed: int = 42) -> Tuple[List[str], List[str]]:
    """(train_ids, val_ids). ``heldout_path`` (``{"asset_ids": [...]}``) holds those ids out as val;
    otherwise a seeded fractional split. ``index`` is the parsed index.json (or its raw_root)."""
    if isinstance(index, str):
        index = load_index(index)
    ids = [e["id"] for e in index["episodes"]]
    if heldout_path:
        with open(heldout_path) as f:
            held = set(json.load(f)["asset_ids"])
        return [i for i in ids if i not in held], [i for i in ids if i in held]
    order = list(ids)
    random.Random(seed).shuffle(order)
    n_val = max(1, int(round(len(order) * val_ratio))) if order else 0
    return order[n_val:], order[:n_val]


class FrameDataset(Dataset):
    """Map-style dataset of T=1 frame windows; ``__getitem__`` -> ``{"obs": {view: uint8 (1,3,H,W)}}``.

    ``random_view``: decode ONE of ``views`` per sample (Python ``random``), returned under "top".
    Otherwise every view in ``views`` is returned under its own key (``all_views`` = per-view val).
    ``hflip_prob``: one flip coin per sample (``torch.rand``), applied to every emitted view.
    ``frame_stride``: index one window every N frames of each episode (1 = every frame)."""

    def __init__(self, raw_root: str, episode_ids: Sequence[str], resolution: int = 256,
                 views: Sequence[str] = VIEWS, random_view: bool = True, hflip_prob: float = 0.5,
                 all_views: bool = False, frame_stride: int = 1, cache_size: int = 256):
        assert not (random_view and all_views), "random_view and all_views are mutually exclusive"
        self.raw_root = raw_root
        self.views = tuple(views)
        self.random_view = bool(random_view)
        self.hflip_prob = float(hflip_prob)
        self.resize = (int(resolution), int(resolution))
        sizes = {e["id"]: int(e["size"]) for e in load_index(raw_root)["episodes"]}
        self.episodes = [(eid, sizes[eid]) for eid in episode_ids]
        stride = max(1, int(frame_stride))
        rows = []
        for ei, (_, size) in enumerate(self.episodes):
            starts = np.arange(0, size, stride, dtype=np.int32)
            if starts.size:
                rows.append(np.stack([np.full_like(starts, ei), starts], axis=1))
        self._index = np.concatenate(rows, axis=0) if rows else np.zeros((0, 2), dtype=np.int32)
        # Per-worker LRU of open decoders (created lazily inside the worker; bounded to cache_size episodes).
        self._cache_size = max(1, int(cache_size))
        self._decoder_cache: "OrderedDict[int, Dict[str, VideoReader]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._index)

    def _decoders(self, ei: int) -> Dict[str, VideoReader]:
        decs = self._decoder_cache.get(ei)
        if decs is not None:
            self._decoder_cache.move_to_end(ei)
            return decs
        eid = self.episodes[ei][0]
        decs = {v: VideoReader(os.path.join(self.raw_root, eid, f"{v}.mp4")) for v in self.views}
        self._decoder_cache[ei] = decs
        if len(self._decoder_cache) > self._cache_size:
            for r in self._decoder_cache.popitem(last=False)[1].values():
                r.close()
        return decs

    def _decode_at(self, dec: VideoReader, indices: List[int]) -> torch.Tensor:
        """Decode + letterbox frames at the given indices -> (N,H,W,C) uint8."""
        return _letterbox_resize(dec.frames(indices), self.resize)

    def __getitem__(self, i: int) -> dict:
        ei, start = (int(v) for v in self._index[i])
        decs = self._decoders(ei)
        if self.random_view:
            cam = random.choice([c for c in self.views if c in decs])
            imgs = {"top": self._decode_at(decs[cam], [start])}
        else:
            imgs = {c: self._decode_at(decs[c], [start]) for c in self.views}
        # One flip coin per SAMPLE, shared by every view (per-worker torch RNG, reseeded each epoch).
        flip = self.hflip_prob > 0.0 and float(torch.rand(())) < self.hflip_prob
        obs = {cam: v.permute(0, 3, 1, 2).contiguous() for cam, v in imgs.items()}   # (1,C,H,W) uint8
        if flip:
            obs = {cam: torch.flip(v, dims=[-1]).contiguous() for cam, v in obs.items()}
        return {"obs": obs}


def make_frame_loader(ds: FrameDataset, batch_size: int, num_workers: int, shuffle: bool,
                      drop_last: bool = True) -> DataLoader:
    """Default collate -> ``batch["obs"][view]`` of shape (B,1,3,H,W) uint8."""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        multiprocessing_context=("fork" if num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        pin_memory=True,
    )
