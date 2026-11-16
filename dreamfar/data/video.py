"""MP4 frame decoding (PyAV, bundled FFmpeg) + the aspect-preserving letterbox resize used for every frame."""
from __future__ import annotations

from typing import Sequence

import av
import numpy as np
import torch
import torch.nn.functional as F


class VideoReader:
    """Random access to the frames of one MP4 by index (constant frame rate), decoded to RGB uint8."""

    def __init__(self, path: str):
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self._pos = 0                      # index of the next frame the decoder will yield
        self._iter = None

    def __len__(self) -> int:
        return int(self.stream.frames)

    def _seek(self, index: int) -> None:
        tb = self.stream.time_base
        ts = int(index / (self.stream.average_rate * tb))
        self.container.seek(ts, stream=self.stream, backward=True, any_frame=False)
        self._iter = self.container.decode(self.stream)
        self._pos = None                   # unknown until the first decoded frame

    def frame(self, index: int) -> np.ndarray:
        """Frame `index` -> (H, W, 3) uint8 RGB."""
        if self._iter is None or self._pos is None or index < self._pos or index > self._pos + 64:
            self._seek(index)
        for fr in self._iter:
            pos = int(round(float(fr.pts * self.stream.time_base) * float(self.stream.average_rate)))
            self._pos = pos + 1
            if pos >= index:
                return fr.to_ndarray(format="rgb24")
        raise IndexError(f"frame {index} past the end of the video")

    def frames(self, indices: Sequence[int]) -> torch.Tensor:
        """Frames at the given indices -> (n, H, W, 3) uint8."""
        return torch.from_numpy(np.stack([self.frame(int(i)) for i in indices]))

    def close(self) -> None:
        self.container.close()


def decode_frames(path: str, indices: Sequence[int]) -> torch.Tensor:
    """Frames at the given native indices of an MP4 -> (n, H, W, 3) uint8."""
    r = VideoReader(path)
    try:
        return r.frames(indices)
    finally:
        r.close()


def letterbox_resize(frames: torch.Tensor, hw: tuple) -> torch.Tensor:
    """Aspect-preserving resize + center zero-pad of a (S,H,W,C) uint8 tensor to (H_out,W_out)."""
    H_out, W_out = hw
    S, H, W, C = frames.shape
    if (H, W) == (H_out, W_out):
        return frames
    scale = min(W_out / W, H_out / H)
    new_w, new_h = max(1, round(W * scale)), max(1, round(H * scale))
    x = frames.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    out = torch.zeros((S, C, H_out, W_out), dtype=x.dtype)
    top = (H_out - new_h) // 2
    left = (W_out - new_w) // 2
    out[:, :, top:top + new_h, left:left + new_w] = x
    return out.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
