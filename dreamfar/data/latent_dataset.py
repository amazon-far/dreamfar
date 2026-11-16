"""Map-style dataset over precomputed latent shards (``dreamfar.data.precompute_latents``).

A shard dir holds, row k = frame k of the concatenated episode stream:
  latents.bin             (N, V, C, h, w) float16     tokenizer latents, views (top, left, right)
  states_actions.bin      (N, 28)         float32     14-D joint  [:, :14]=state [:, 14:]=action
  states_actions_ee.bin   (N, 40)         float32     20-D EE pose [:, :20]=state [:, 20:]=action
  index.json              {views, latent_shape, latent_dtype, fps, episodes:[{asset_id,start,size,task_name}]}

The model is CONDITIONED on the 20-D EE action and PREDICTS the 14-D joint state; keyframe retrieval
keys are the 20-D EE state, queried by the EE action.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .contact import sidecar_to_dense

ACTION_BIN, ACTION_COLS = "states_actions_ee.bin", 40    # action (conditioning) = 20-D EE pose
STATE_BIN, STATE_COLS = "states_actions.bin", 28         # state token = 14-D joints


@dataclass
class _LatentEpisode:
    asset_id: str
    shard_dir: str
    start: int          # row offset of frame 0 within this shard's .bin
    size: int           # frame count
    task_name: str = ""


def _open_sa_memmap(path: str, cols: int) -> np.memmap:
    """Open a shard state/action bin as float32 (rows, cols), validating the byte length."""
    size = os.path.getsize(path)
    stride = cols * 4
    if size % stride:
        raise ValueError(f"{path}: {size} bytes is not a whole number of float32 rows of {cols} cols")
    return np.memmap(path, dtype=np.float32, mode="r").reshape(size // stride, cols)


# --------------------------------------------------------------------------- #
# Retrieval scoring (shared verbatim with inference)
# --------------------------------------------------------------------------- #
def arm_dim_groups(state_dim: int) -> tuple:
    """(left_arm, right_arm, grippers, no_gripper) dim lists for [left block | right block], each block
    state_dim//2 wide and ending in its gripper: 14-D -> left 0..5, 6, right 7..12, 13; 20-D -> 0..8, 9, 10..18, 19."""
    half = state_dim // 2
    left = list(range(0, half - 1))
    right = list(range(half, 2 * half - 1))
    grippers = [half - 1, 2 * half - 1]
    return left, right, grippers, left + right


def keyframe_similarity(cand_states: np.ndarray, queries: np.ndarray, *, reduce_n: str = "mean") -> np.ndarray:
    """Pose similarity of (m, D) candidates to (n, D) queries: per-arm L2 over the arm's pose dims
    (grippers excluded), keep the CLOSER arm, negated so higher = closer. ``reduce_n``: "mean" | "max"
    over the n queries, or "none" for the unreduced (m, n) matrix."""
    if queries.ndim == 1:
        queries = queries[None, :]
    left, right, _grippers, _no_gripper = arm_dim_groups(cand_states.shape[1])
    dl = np.linalg.norm(cand_states[:, None, left] - queries[None, :, left], axis=2)
    dr = np.linalg.norm(cand_states[:, None, right] - queries[None, :, right], axis=2)
    per_query = np.maximum(-dl, -dr)
    if reduce_n == "none":
        return per_query                                                 # (m,n)
    return per_query.mean(axis=1) if reduce_n == "mean" else per_query.max(axis=1)


def keyframe_confidence(cand_states: np.ndarray, queries: np.ndarray, bank: np.ndarray,
                        win_start: int, *, sim_weight: float, time_weight: float,
                        reduce_n: str = "mean", normalize_sim: bool = False,
                        dt_ref: Optional[float] = None) -> np.ndarray:
    """Retrieval confidence = sim_weight * sim - time_weight * recency. ``normalize_sim`` min-maxes the sim
    term into [0,1] per query; recency = min(|win_start - bank| / dt_ref, 1) in NATIVE frames
    (``dt_ref=None``: |dt| / win_start)."""
    sim = keyframe_similarity(cand_states, queries, reduce_n=reduce_n)
    if normalize_sim:
        if sim.ndim == 2:            # per-QUERY min-max: each column keeps its own window scaling
            lo, hi = sim.min(axis=0, keepdims=True), sim.max(axis=0, keepdims=True)
            sim = np.where(hi - lo > 1e-12, (sim - lo) / np.maximum(hi - lo, 1e-12), 0.0)
        else:
            lo, hi = float(sim.min()), float(sim.max())
            sim = (sim - lo) / (hi - lo) if (hi - lo) > 1e-12 else np.zeros_like(sim)
    if dt_ref is not None and dt_ref > 0:
        tdist = np.minimum(np.abs(win_start - bank).astype(np.float32) / float(dt_ref), 1.0)
    else:
        tdist = np.abs(win_start - bank).astype(np.float32) / max(win_start, 1)
    if sim.ndim == 2:
        return sim_weight * sim - time_weight * tdist[:, None]
    return sim_weight * sim - time_weight * tdist


def vote_topk(conf: np.ndarray, k: int) -> np.ndarray:
    """Hard vote over per-query rankings: ``conf`` (m candidates, n queries) -> the k most-voted row
    indices (each query votes for its own top-k; ties break on summed confidence scaled into [0, 0.5])."""
    m, n = conf.shape
    k = min(int(k), m)
    if k >= m:
        return np.arange(m)
    top = np.argpartition(-conf, k - 1, axis=0)[:k]                      # (k,n) rows per query
    votes = np.bincount(top.ravel(), minlength=m).astype(np.float64)
    s = conf.sum(axis=1)
    rng = float(s.max() - s.min())
    tie = (s - s.min()) / rng * 0.5 if rng > 1e-12 else np.zeros_like(s)
    return np.argpartition(-(votes + tie), k - 1)[:k]


# --------------------------------------------------------------------------- #
# Catalog + split
# --------------------------------------------------------------------------- #
def load_catalog(latent_root: str) -> Tuple[List[_LatentEpisode], dict]:
    """Read every ``<latent_root>/shard_*/index.json`` -> (episodes sorted by asset_id, meta).
    ``meta`` = {views, latent_shape:[V,C,h,w], latent_dtype, fps, resolution}."""
    shard_dirs = sorted(
        os.path.join(latent_root, d) for d in os.listdir(latent_root)
        if d.startswith("shard_") and os.path.isdir(os.path.join(latent_root, d))
    )
    if not shard_dirs:
        raise FileNotFoundError(f"no shard_* dirs under {latent_root}")
    eps: List[_LatentEpisode] = []
    meta = None
    for sd in shard_dirs:
        idx_path = os.path.join(sd, "index.json")
        if not os.path.exists(idx_path):
            continue
        idx = json.load(open(idx_path))
        if meta is None:
            meta = {"views": idx["views"], "latent_shape": idx["latent_shape"],
                    "latent_dtype": idx.get("latent_dtype", "float16"),
                    "fps": idx.get("fps", 30), "resolution": idx.get("resolution")}
        for e in idx["episodes"]:
            eps.append(_LatentEpisode(e["asset_id"], sd, int(e["start"]), int(e["size"]),
                                      e.get("task_name", "")))
    if meta is None:
        raise FileNotFoundError(f"no index.json found under {latent_root}")
    eps.sort(key=lambda e: e.asset_id)
    return eps, meta


def split_heldout(episodes: Sequence[_LatentEpisode], heldout_path: str) -> Tuple[list, list]:
    """(train, val) by the held-out asset_id set ``{"asset_ids": [...]}``."""
    ids = set(json.load(open(heldout_path))["asset_ids"])
    train = [e for e in episodes if e.asset_id not in ids]
    val = [e for e in episodes if e.asset_id in ids]
    return train, val


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class LatentShardDataset(Dataset):
    """Windows of ``sequence_length`` model frames (``frame_skip`` native frames apart) over latent shards,
    with per-frame contact GT, optional MCP future targets and per-frame retrieved keyframe memory.

    Train: window_stride=20, random_start_jitter=True, n_memory=8, stochastic_select=True, mcp_target_frames=3.
    Val:   window_stride=1, random_start_jitter=False, n_memory=0, mcp_target_frames=0 (deterministic).
    """

    def __init__(
        self,
        episodes: Sequence[_LatentEpisode],
        latent_shape: Sequence[int],     # (V, C, h, w)
        sequence_length: int,
        contact_labels_dir: Optional[str],
        *,
        latent_dtype: str = "float16",
        window_stride: int = 1,
        frame_skip: int = 1,
        random_start_jitter: bool = True,
        contact_min_dwell: int = 0,
        n_memory: int = 0,
        n_mem_pool: int = 30,
        mem_dropout_prob: float = 0.2,
        stochastic_select: bool = False,
        kf_sim_weight: float = 1.0,
        kf_time_weight: float = 0.2,
        kf_normalize_sim: bool = True,
        kf_dt_ref_s: float = 30.0,
        fps: float = 30.0,
        mcp_target_frames: int = 0,
    ):
        self.V, self.C, self.h, self.w = map(int, latent_shape)
        self.S = int(sequence_length)
        self.lat_dtype = np.float16 if latent_dtype == "float16" else np.float32
        self.action_dim = ACTION_COLS // 2
        self.state_dim = STATE_COLS // 2
        self.window_stride = max(1, int(window_stride))
        self.frame_skip = max(1, int(frame_skip))
        # with window_stride>1 each sample's start is jittered by U{0..stride-1} at fetch time (clamped
        # in-episode) so the frames the fixed grid skips are covered across epochs
        self.random_start_jitter = bool(random_start_jitter) and self.window_stride > 1
        # None = no contact sidecars (inference without GT contact; memory selection then needs n_memory=0)
        self.contact_labels_dir = None if contact_labels_dir is None else str(contact_labels_dir)
        self.contact_min_dwell = int(contact_min_dwell)
        # STREAMING per-frame memory: slot 0 = anchor (random early frame at train, frame 0 otherwise),
        # slots 1..P-1 = up to n_mem_pool-1 past contact keyframes; each context frame is masked to the
        # anchor + its own top-(n_memory-1) by retrieval score, then per-(frame,slot) dropout (train only).
        self.n_memory = max(0, int(n_memory))
        self.n_mem_pool = max(int(n_mem_pool), int(n_memory))
        self.mem_dropout_prob = float(mem_dropout_prob)
        self.stochastic_select = bool(stochastic_select)
        self.kf_sim_weight = float(kf_sim_weight)
        self.kf_time_weight = float(kf_time_weight)
        self.kf_normalize_sim = bool(kf_normalize_sim)
        self.fps = float(fps)
        self.kf_dt_ref = float(kf_dt_ref_s) * self.fps if float(kf_dt_ref_s) > 0 else None   # native frames
        # MCP: emit `mcp_target_frames` extra model-frames past the window as TARGETS ONLY
        self.mcp_target_frames = int(mcp_target_frames)
        assert self.n_memory == 0 or self.contact_labels_dir is not None, "memory retrieval needs contact sidecars"
        # episodes without a contact sidecar cannot be supervised -> dropped
        self.episodes = [e for e in episodes
                         if self.contact_labels_dir is None or os.path.exists(self._contact_path(e))]
        # per-worker caches: full-episode arrays (read once per episode) + lazy memmaps per shard dir
        self._state_cache: dict = {}     # asset_id -> (size, 14) joint state
        self._act_cache: dict = {}       # asset_id -> (size, 20) EE action
        self._eest_cache: dict = {}      # asset_id -> (size, 20) EE state (retrieval keys)
        self._contact_cache: dict = {}   # asset_id -> {"kf": int64[], "cl": int8[size], "cr": int8[size]}
        self._lat_mm: dict = {}
        self._sa_mm: dict = {}
        self._state_bin_mm: dict = {}
        self._index: List[tuple] = []
        self.build_index()

    # -- window index --------------------------------------------------------- #
    def build_index(self) -> "LatentShardDataset":
        self._index.clear()
        span = (self.S - 1) * self.frame_skip + 1
        for ei, ep in enumerate(self.episodes):
            n_windows = ep.size - (span - 1)
            if n_windows <= 0:
                continue
            for s in range(0, n_windows, self.window_stride):
                self._index.append((ei, s))
        return self

    def __len__(self) -> int:
        return len(self._index)

    # -- lazy per-worker memmaps ---------------------------------------------- #
    def _latents(self, shard_dir: str) -> np.memmap:
        mm = self._lat_mm.get(shard_dir)
        if mm is None:
            mm = np.memmap(os.path.join(shard_dir, "latents.bin"), dtype=self.lat_dtype,
                           mode="r").reshape(-1, self.V, self.C, self.h, self.w)
            self._lat_mm[shard_dir] = mm
        return mm

    def _sa(self, shard_dir: str) -> np.memmap:
        """The EE bin (40 cols): action = cols [20:], EE state = cols [:20]."""
        mm = self._sa_mm.get(shard_dir)
        if mm is None:
            mm = _open_sa_memmap(os.path.join(shard_dir, ACTION_BIN), ACTION_COLS)
            self._sa_mm[shard_dir] = mm
        return mm

    def _state_mm(self, shard_dir: str) -> np.memmap:
        """The joint bin (28 cols): state = cols [:14]."""
        mm = self._state_bin_mm.get(shard_dir)
        if mm is None:
            mm = _open_sa_memmap(os.path.join(shard_dir, STATE_BIN), STATE_COLS)
            self._state_bin_mm[shard_dir] = mm
        return mm

    def _gather(self, shard_dir: str, base: int, idx: Sequence[int],
                act_idx: Optional[Sequence[int]] = None) -> tuple:
        """(latent (n,V,C,h,w), action (n,20), state (n,14)) at rows base+idx; ``act_idx`` reads the
        action at other rows (MCP extension) while latents/state stay at ``idx``."""
        A = self.action_dim
        rows = [base + i for i in idx]
        lat = np.ascontiguousarray(self._latents(shard_dir)[rows]).astype(np.float32)   # (n,V,C,h,w)
        sa = np.ascontiguousarray(self._sa(shard_dir)[rows]).astype(np.float32)          # (n,40)
        latt = torch.from_numpy(lat)
        if act_idx is None:
            act = torch.from_numpy(np.ascontiguousarray(sa[:, A:]))                      # action(A)
        else:
            arows = [base + i for i in act_idx]
            asa = np.ascontiguousarray(self._sa(shard_dir)[arows]).astype(np.float32)
            act = torch.from_numpy(np.ascontiguousarray(asa[:, A:]))                     # action(A)
        sb = self._state_mm(shard_dir)[rows]
        state = torch.from_numpy(np.ascontiguousarray(
            np.asarray(sb, dtype=np.float32)[:, :self.state_dim]))                       # state(D)
        return latt, act, state

    def _episode_state(self, ep) -> np.ndarray:
        """Full-episode joint state (size,14), cached per worker."""
        st = self._state_cache.get(ep.asset_id)
        if st is None:
            rows = [ep.start + i for i in range(ep.size)]
            sb = self._state_mm(ep.shard_dir)[rows]
            st = np.ascontiguousarray(np.asarray(sb, dtype=np.float32)[:, :self.state_dim]).astype(np.float32)
            self._state_cache[ep.asset_id] = st
        return st

    def _episode_action(self, ep) -> np.ndarray:
        """Full-episode commanded EE action (size,20) = EE bin cols [20:40], cached per worker."""
        a = self._act_cache.get(ep.asset_id)
        if a is None:
            A = self.action_dim
            rows = [ep.start + i for i in range(ep.size)]
            sa = self._sa(ep.shard_dir)[rows]
            a = np.ascontiguousarray(sa[:, A:2 * A]).astype(np.float32)      # action(A)
            self._act_cache[ep.asset_id] = a
        return a

    def _episode_action_space_state(self, ep) -> np.ndarray:
        """Full-episode EE state (size,20) = EE bin cols [:20]: the retrieval KEY space."""
        k = self._eest_cache.get(ep.asset_id)
        if k is None:
            A = self.action_dim
            rows = [ep.start + i for i in range(ep.size)]
            k = np.ascontiguousarray(self._sa(ep.shard_dir)[rows][:, :A]).astype(np.float32)
            self._eest_cache[ep.asset_id] = k
        return k

    def _contact_path(self, ep) -> str:
        """<contact_labels_dir>/<asset_id with non-alphanumerics -> '_'>.json"""
        safe = "".join(c if c.isalnum() else "_" for c in str(ep.asset_id))
        return os.path.join(self.contact_labels_dir, safe + ".json")

    def _episode_contact(self, ep) -> dict:
        """Contact sidecar as dense per-native-frame arrays {"kf", "cl", "cr"} (see ``sidecar_to_dense``)."""
        c = self._contact_cache.get(ep.asset_id)
        if c is None:
            c = sidecar_to_dense(json.load(open(self._contact_path(ep))), ep.size, self.contact_min_dwell)
            self._contact_cache[ep.asset_id] = c
        return c

    def _episode_keyframes(self, ep) -> np.ndarray:
        return self._episode_contact(ep)["kf"]

    def _select_memory_stream_per_frame(self, ep, win_start: int, want_idx) -> tuple:
        """Per-frame streaming memory selection. Returns (idx[P], frame_mask[S,P], dt[P], n_real).

        Bank = contact keyframes <= win_start. Slot 0 = anchor (random frame <= min(bank) at train, else 0);
        the rest = up to P-1 bank frames (random subsample if more). Context frame j is masked to the anchor
        + its top-(n_memory-1) candidates under ``keyframe_confidence`` with query = the frame's cond action
        (native row want_idx[j]-1) against the EE-state keys; then per-(frame,slot) dropout (train only).
        ``dt`` is in MODEL frames from want_idx[0]."""
        P = int(self.n_mem_pool)
        K = int(self.n_memory)
        S = len(want_idx)
        fs = float(self.frame_skip)
        kf = np.asarray(self._episode_keyframes(ep), dtype=np.int64)
        bank = kf[kf <= win_start]
        keys = self._episode_action_space_state(ep)
        qsrc = self._episode_action(ep)
        qrows = [max(0, int(wi) - 1) for wi in want_idx]
        nrows = keys.shape[0]
        hi = int(bank.min()) if bank.size else int(win_start)
        anchor = random.randint(0, hi) if (self.stochastic_select and hi > 0) else 0
        cand = bank[bank != anchor]
        if cand.size > P - 1:
            cand = np.sort(np.random.choice(cand, size=P - 1, replace=False))
        slots = np.concatenate([np.array([anchor], dtype=np.int64), cand.astype(np.int64)])
        p_real = int(slots.size)
        idx = [int(f) for f in slots] + [0] * (P - p_real)     # padded slot content is masked out
        frame_mask = np.zeros((S, P), dtype=bool)
        frame_mask[:, 0] = True                                # anchor -> every frame (pre-dropout)
        surv = list(range(1, p_real))                          # ALL real non-anchor slots are candidates
        if surv:
            sl = np.asarray([idx[i] for i in surv], dtype=np.int64)
            prow = np.clip(sl, 0, nrows - 1)
            q = qsrc[np.clip(np.asarray(qrows, dtype=np.int64), 0, qsrc.shape[0] - 1)]
            conf = keyframe_confidence(
                keys[prow], q, sl, max(win_start, 1),
                sim_weight=self.kf_sim_weight, time_weight=self.kf_time_weight,
                reduce_n="none", normalize_sim=self.kf_normalize_sim, dt_ref=self.kf_dt_ref)   # (m, S)
            k = min(K - 1, len(surv))
            for j in range(S):
                cj = conf[:, j]
                top = (np.argpartition(-cj, k - 1)[:k] if len(surv) > k
                       else np.arange(len(surv)))
                for t in top:
                    frame_mask[j, surv[int(t)]] = True
        # dropout LAST, per (frame, slot) entry, anchor included
        if self.stochastic_select and self.mem_dropout_prob > 0.0:
            drop = np.random.random_sample(frame_mask.shape) < self.mem_dropout_prob
            frame_mask &= ~drop
        dt = [max(0.0, (win_start - idx[i]) / fs) for i in range(P)]
        return idx, frame_mask, dt, p_real

    # -- sample fetch --------------------------------------------------------- #
    def __getitem__(self, i: int) -> dict:
        ei, start = self._index[i]
        ep = self.episodes[ei]
        S = self.S
        span = (S - 1) * self.frame_skip + 1
        if self.random_start_jitter:
            max_start = ep.size - span
            start = min(max_start, start + random.randint(0, self.window_stride - 1))
        want_idx = list(range(start, start + S * self.frame_skip, self.frame_skip))

        lat, action, state = self._gather(ep.shard_dir, ep.start, want_idx)      # (S,V,C,h,w),(S,20),(S,14)
        out = {"latent": lat, "action": action.float(), "state": state.float(),
               "frame_idx": torch.tensor(want_idx, dtype=torch.float32)}        # (S,) native frames
        if self.contact_labels_dir is not None:
            c = self._episode_contact(ep)
            ci = np.clip(np.asarray(want_idx), 0, ep.size - 1)
            out["contact"] = torch.from_numpy(
                np.stack([c["cl"][ci], c["cr"][ci]], axis=1)).float()            # (S,2) [L,R]

        if self.mcp_target_frames > 0:
            # K extra model-frames PAST the window, targets only. The window's cond is roll(action, +1)
            # and the extension is concatenated after it, so extension row j carries the action of the
            # PREVIOUS model frame (fut[j] - frame_skip). `future_valid` is False past the episode end.
            K = self.mcp_target_frames
            fut = [want_idx[-1] + j * self.frame_skip for j in range(1, K + 1)]
            ok = [f <= ep.size - 1 for f in fut]
            fut_c = [min(f, ep.size - 1) for f in fut]
            a_idx = [f - self.frame_skip for f in fut_c]
            f_act = [int(min(max(a, 0), ep.size - 1)) for a in a_idx]
            fg = self._gather(ep.shard_dir, ep.start, fut_c, act_idx=f_act)
            out["latent_future"] = fg[0]                              # (K,V,C,h,w)
            out["action_future"] = fg[1].float()                      # (K,20)
            out["state_future"] = fg[2].float()                       # (K,14)
            out["future_valid"] = torch.tensor(ok, dtype=torch.bool)  # (K,)

        if self.n_memory > 0:
            mem_idx, fmask, mem_dt, n_real = self._select_memory_stream_per_frame(ep, start, want_idx)
            mg = self._gather(ep.shard_dir, ep.start, mem_idx)
            out["latent_memory"] = mg[0]                              # (P,V,C,h,w)
            out["state_memory"] = mg[2].float()                       # (P,14)
            out["mem_dt"] = torch.tensor(mem_dt, dtype=torch.float32)          # (P,) model-frame dt >= 0
            out["mem_frame_mask"] = torch.from_numpy(fmask)                    # (S,P) bool per-frame visibility
            out["mem_valid"] = torch.from_numpy(fmask.any(axis=0))             # (P,) bool
            out["mem_frame_idx"] = torch.tensor(mem_idx, dtype=torch.float32)  # (P,) native frames
            out["mem_n_real"] = torch.tensor(float(n_real), dtype=torch.float32)
        return out


def make_latent_loader(ds: LatentShardDataset, batch_size: int, num_workers: int, shuffle: bool,
                       seed: int = 0, distributed: bool = False, drop_last: bool = True) -> DataLoader:
    """DataLoader over ``ds`` (default_collate stacks the flat dicts); ``distributed`` attaches a
    ``DistributedSampler`` (call ``loader.sampler.set_epoch(epoch)`` each epoch)."""
    rank = torch.distributed.get_rank() if distributed else 0
    g = torch.Generator()
    g.manual_seed(int(seed) + rank)
    sampler = DistributedSampler(ds, shuffle=shuffle, seed=int(seed), drop_last=drop_last) if distributed else None
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(shuffle and sampler is None), sampler=sampler,
        num_workers=num_workers, drop_last=drop_last, generator=g,
        multiprocessing_context=("fork" if num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
        pin_memory=True,
    )
