"""Contact-label smoothing + keyframe derivation shared by the training loader and inference.

Contact sidecar (one JSON per episode): ``{"frames": [native idx at 10 Hz], "contact": {"left": [0/1..],
"right": [0/1..]}, "keyframes": [native idx]}``.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def median_filter(seq: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1 or seq.size == 0:
        return seq.copy()
    k = window // 2
    pad = np.pad(seq.astype(np.int8), k, mode="edge")
    out = np.empty_like(seq, dtype=np.int8)
    for i in range(seq.size):
        out[i] = 1 if pad[i:i + window].sum() * 2 > window else 0
    return out


def min_dwell(seq: np.ndarray, dwell: int = 3) -> np.ndarray:
    """Flip any run shorter than `dwell` into its (already-cleaned) left neighbor's value."""
    if dwell <= 1 or seq.size == 0:
        return seq.astype(np.int8).copy()
    out = seq.astype(np.int8).copy()
    i = 0
    while i < out.size:
        j = i
        while j < out.size and out[j] == out[i]:
            j += 1
        if j - i < dwell and i > 0:            # short run (keep the leading run as-is)
            out[i:j] = out[i - 1]
        i = j
    return out


def smooth_prob(prob: Dict[int, float], frames: Sequence[int], box_window: int = 3,
                enter: float = 0.6, leave: float = 0.4) -> Dict[int, bool]:
    """Box-average P(contact) over `frames`, then hysteresis-threshold (enter > 0.6, leave < 0.4).
    Returns {frame -> bool}."""
    if not frames:
        return {}
    p = np.array([prob[f] for f in frames], dtype=np.float32)
    if box_window > 1:
        k = box_window // 2
        pad = np.pad(p, k, mode="edge")
        p = np.array([pad[i:i + box_window].mean() for i in range(p.size)], dtype=np.float32)
    out = np.zeros(p.size, dtype=np.int8)
    state = p[0] >= 0.5                                  # seed from the first frame
    for i, pi in enumerate(p):
        if state and pi < leave:
            state = False
        elif (not state) and pi > enter:
            state = True
        out[i] = 1 if state else 0
    return {f: bool(v) for f, v in zip(frames, out)}


def derive_keyframes(contact_by_arm: Dict[str, Dict[int, bool]], frames: Sequence[int],
                     min_gap: int = 2) -> List[int]:
    """Keyframe = any frame where EITHER arm's contact FLIPS; merge within `min_gap` grid steps.
    Transitions only -- frame 0 is NOT added here."""
    if not frames:
        return []
    kf = set()
    for arm in contact_by_arm:
        prev = None
        for f in frames:
            if prev is not None and contact_by_arm[arm][f] != contact_by_arm[arm][prev]:
                kf.add(f)
            prev = f
    pos = {f: i for i, f in enumerate(frames)}
    kept: List[int] = []
    for f in sorted(kf):
        if not kept or pos[f] - pos[kept[-1]] >= min_gap:
            kept.append(f)
    return kept


def dwell_filter_keyframes(bits_by_arm: Dict[str, np.ndarray], frames: Sequence[int],
                           dwell: int) -> tuple:
    """Per-arm ``min_dwell(dwell)`` over the `frames` grid, then ``derive_keyframes`` on the filtered
    bits. Returns ({arm: int8 bits aligned to `frames`}, sorted keyframe list in native indices)."""
    out = {arm: min_dwell(np.asarray(b, dtype=np.int8), int(dwell)) for arm, b in bits_by_arm.items()}
    kf = derive_keyframes({arm: {f: bool(v) for f, v in zip(frames, out[arm])} for arm in out}, frames)
    return out, kf


class ContactKeyframeTracker:
    """Incremental twin of ``smooth_prob -> min_dwell -> derive_keyframes(min_gap=2) -> | {0}`` over a
    growing prefix: bit-identical to re-running the batch pipeline over all ingested frames, at O(1)
    amortized cost per frame. Frames are ingested in increasing order with one probability per arm."""

    def __init__(self, n_arms: int = 2, dwell: int = 0, enter: float = 0.6, leave: float = 0.4,
                 min_gap: int = 2, box_window: int = 3):
        assert box_window == 3, "only the box-3 smoothing of smooth_prob is mirrored"
        self.A, self.dwell, self.enter, self.leave, self.min_gap = int(n_arms), int(dwell), enter, leave, min_gap
        self.frames: List[int] = []                            # ingested frame ids, ascending
        self._raw = [[] for _ in range(self.A)]                # np.float32 raw probs per arm
        self._hst = [False] * self.A                           # hysteresis state after the last FINAL index
        self._nfinal = 0                                       # indices < this have final hysteresis bits
        # open run over the FINAL bits (min_dwell bookkeeping) + cleaned bits of the CLOSED runs
        self._rs, self._rl, self._rv = [0] * self.A, [0] * self.A, [0] * self.A
        self._cvp = [0] * self.A                               # cleaned value of the run before the open one
        self._cb = [bytearray() for _ in range(self.A)]        # final cleaned bits (closed runs only)
        self._kept: List[int] = []                             # final kept flip INDICES (greedy min-gap)
        self._nflip = 1                                        # flips at indices < this are final
        self._bank = np.zeros(1, dtype=np.int64)               # [0] + frames of `_kept`, rebuilt on change
        self._next_frame = 0                                   # `advance` cursor

    # ---- exact replicas of smooth_prob's arithmetic ------------------------------------------------
    def _sm(self, a: int, i: int, n: int):
        r = self._raw[a]
        return np.array([r[max(i - 1, 0)], r[i], r[min(i + 1, n - 1)]], dtype=np.float32).mean()

    def _hyst(self, state: bool, pi) -> bool:
        if state and pi < self.leave:
            return False
        if (not state) and pi > self.enter:
            return True
        return state

    def _bit(self, a: int, i: int, n: int) -> int:
        sm = self._sm(a, i, n)
        st = bool(sm >= 0.5) if i == 0 else self._hst[a]
        st = self._hyst(st, sm)
        if i == self._nfinal and i <= n - 2:                  # finalize (smoothed value no longer moves)
            self._hst[a] = st
        return 1 if st else 0

    def _cv(self, a: int, rs: int, rl: int, rv: int) -> int:
        """min_dwell's rule: a run keeps its value if it is the first run or >= dwell long."""
        return rv if (rs == 0 or rl >= self.dwell) else self._cvp[a]

    # ---- ingestion ---------------------------------------------------------------------------------
    def ingest(self, frame: int, probs) -> None:
        assert not self.frames or frame > self.frames[-1], "frames must be ingested in increasing order"
        self.frames.append(int(frame))
        for a in range(self.A):
            self._raw[a].append(np.float32(probs[a]))
        n = len(self.frames)
        if n < 2:
            return
        i = n - 2                                              # index whose smoothed value just became final
        for a in range(self.A):
            b = self._bit(a, i, n)
            if i == 0:
                self._rs[a], self._rl[a], self._rv[a] = 0, 1, b
            elif b == self._rv[a]:
                self._rl[a] += 1
            else:                                              # the open run closes -> its cleaned value is final
                cv = self._cv(a, self._rs[a], self._rl[a], self._rv[a])
                self._cb[a].extend(bytes([cv]) * self._rl[a])
                self._cvp[a] = cv
                self._rs[a], self._rl[a], self._rv[a] = i, 1, b
        self._nfinal = n - 1

    def advance(self, lr: dict, upto: int) -> np.ndarray:
        """Ingest every frame in [cursor, upto) present in `lr` (frame -> per-arm probs), then return
        the keyframe bank for frames [0, upto)."""
        for f in range(self._next_frame, upto):
            if f in lr:
                self.ingest(f, lr[f])
        self._next_frame = max(self._next_frame, upto)
        return self.keyframes()

    # ---- query -------------------------------------------------------------------------------------
    def keyframes(self) -> np.ndarray:
        n = len(self.frames)
        if n == 0:
            return np.zeros(1, dtype=np.int64)
        # provisional cleaned bits of the tail [settled, n) per arm; settled = where the open run's cleaned
        # value is still undecided (dwell) -- otherwise n-1 (only the last frame's smoothed value moves)
        tails, settled = [], n - 1
        for a in range(self.A):
            b = self._bit(a, n - 1, n)                         # provisional bit of the newest frame
            if n == 1:
                tails.append((0, [b]))
                settled = 0
                continue
            rs, rl, rv = self._rs[a], self._rl[a], self._rv[a]
            if b == rv:
                cv_open = self._cv(a, rs, rl + 1, rv)
                tail = [cv_open] * (n - rs)
            else:
                cv_open = self._cv(a, rs, rl, rv)
                cv_new = b if 1 >= self.dwell else cv_open       # a 1-long new run
                tail = [cv_open] * (n - 1 - rs) + [cv_new]
            s_a = rs if (rs > 0 and rl < self.dwell) else n - 1
            tails.append((rs, tail))
            settled = min(settled, s_a)

        def cb(a: int, i: int) -> int:                        # cleaned bit of arm a at index i (any i < n)
            rs, tail = tails[a]
            if i >= rs:
                return tail[i - rs]
            return self._cb[a][i]                             # closed runs cover [0, rs)

        def flip(i: int) -> bool:
            return any(cb(a, i) != cb(a, i - 1) for a in range(self.A))

        n_kept = len(self._kept)
        for i in range(self._nflip, settled):                 # finalize flips below the settled index
            if flip(i) and (not self._kept or i - self._kept[-1] >= self.min_gap):
                self._kept.append(i)
        self._nflip = max(self._nflip, settled)
        if len(self._kept) != n_kept:                         # the final bank changed (once per keyframe)
            self._bank = np.asarray([0] + [self.frames[i] for i in self._kept], dtype=np.int64)
        last = self._kept[-1] if self._kept else None
        tail: List[int] = []
        for i in range(max(settled, 1), n):                   # provisional tail (<= dwell + 1 indices)
            if flip(i) and (last is None or i - last >= self.min_gap):
                tail.append(self.frames[i])
                last = i
        return np.concatenate([self._bank, np.asarray(tail, dtype=np.int64)]) if tail else self._bank.copy()


def sidecar_to_dense(r: dict, size: int, min_dwell: int = 0) -> dict:
    """Contact sidecar -> dense per-NATIVE-frame arrays: the 10-Hz labels are held forward over
    [0, size); keyframes = transitions only (frame 0 stripped). ``min_dwell >= 1`` de-combs the bits
    first and re-derives the keyframes. Returns {"kf": int64[], "cl": int8[size], "cr": int8[size]}."""
    frames = np.asarray(r["frames"], dtype=np.int64)
    cl = np.asarray(r["contact"]["left"], dtype=np.int8)
    cr = np.asarray(r["contact"]["right"], dtype=np.int8)
    kf_list = r["keyframes"]
    if min_dwell >= 1 and frames.size:
        bits, kf_list = dwell_filter_keyframes({"left": cl, "right": cr}, frames.tolist(), min_dwell)
        cl, cr = bits["left"], bits["right"]
    dl = np.zeros(size, dtype=np.int8); dr = np.zeros(size, dtype=np.int8)
    # nearest-label = forward-fill from the labeled 10Hz frames across all native indices.
    if frames.size:
        idx = np.searchsorted(frames, np.arange(size), side="right") - 1
        idx = np.clip(idx, 0, frames.size - 1)
        dl = cl[idx]; dr = cr[idx]
    kf = np.asarray([k for k in kf_list if k != 0], dtype=np.int64)
    return {"kf": kf, "cl": dl, "cr": dr}
