"""Per-key affine normalizer (x -> x * scale + offset) whose state_dict travels inside checkpoints.

State-dict layout is ``params_dict.<key>.{scale,offset,input_stats.{min,max,mean,std}}``.
"""
from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn

ArrayLike = Union[torch.Tensor, np.ndarray]


def _to_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.flatten()


class SingleFieldLinearNormalizer(nn.Module):
    """Affine normalizer for one field, parameterised by (scale, offset)."""

    def __init__(self, params_dict: nn.ParameterDict | None = None):
        super().__init__()
        self.params_dict = params_dict if params_dict is not None else nn.ParameterDict()

    @classmethod
    def create_manual(cls, scale: ArrayLike, offset: ArrayLike, input_stats: Dict[str, ArrayLike]):
        pd = nn.ParameterDict(
            {
                "scale": nn.Parameter(_to_tensor(scale), requires_grad=False),
                "offset": nn.Parameter(_to_tensor(offset), requires_grad=False),
                "input_stats": nn.ParameterDict(
                    {k: nn.Parameter(_to_tensor(v), requires_grad=False) for k, v in input_stats.items()}
                ),
            }
        )
        return cls(pd)

    def _apply_affine(self, x: ArrayLike, forward: bool) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        scale = self.params_dict["scale"]
        offset = self.params_dict["offset"]
        x = x.to(device=scale.device, dtype=scale.dtype)
        src_shape = x.shape
        x = x.reshape(-1, scale.shape[0])
        x = x * scale + offset if forward else (x - offset) / scale
        return x.reshape(src_shape)

    def normalize(self, x: ArrayLike) -> torch.Tensor:
        return self._apply_affine(x, forward=True)

    def unnormalize(self, x: ArrayLike) -> torch.Tensor:
        return self._apply_affine(x, forward=False)


class LinearNormalizer(nn.Module):
    """Dict of per-key affine normalizers; round-trips via ``state_dict``."""

    def __init__(self):
        super().__init__()
        self.params_dict = nn.ParameterDict()

    def __setitem__(self, key: str, value: SingleFieldLinearNormalizer) -> None:
        self.params_dict[key] = value.params_dict

    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __contains__(self, key: str) -> bool:
        return key in self.params_dict

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Rebuild the (dynamic) ParameterDict tree from the flat keys under ``<prefix>params_dict``.
        def dfs_add(dest: nn.ParameterDict, keys: list, value: torch.Tensor) -> None:
            if len(keys) == 1:
                dest[keys[0]] = nn.Parameter(value.clone(), requires_grad=False)
                return
            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        out = nn.ParameterDict()
        root = prefix + "params_dict"
        for key, value in state_dict.items():
            if key.startswith(root):
                param_keys = key[len(root):].split(".")[1:]
                if param_keys:
                    dfs_add(out, param_keys, value)
        self.params_dict = out
        self.params_dict.requires_grad_(False)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        self._load_from_state_dict(state_dict, "", {}, strict, [], [], [])
        return torch.nn.modules.module._IncompatibleKeys([], [])


def get_bounds_normalizer(
    lo: np.ndarray, hi: np.ndarray, output_max: float = 1, output_min: float = -1
) -> SingleFieldLinearNormalizer:
    """Map a KNOWN per-dim bound [lo, hi] to [output_min, output_max] (lo=-1, hi=1 is the exact identity)."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if np.any(hi <= lo):
        raise ValueError(f"need hi > lo everywhere; bad dims {np.nonzero(hi <= lo)[0]}")
    scale = (output_max - output_min) / (hi - lo)
    offset = output_min - scale * lo
    stat = {"min": lo, "max": hi, "mean": (lo + hi) / 2, "std": (hi - lo) / np.sqrt(12.0)}
    return SingleFieldLinearNormalizer.create_manual(
        scale.astype(np.float32), offset.astype(np.float32),
        {k: v.astype(np.float32) for k, v in stat.items()}
    )


def load_normalizer_file(path: str) -> "LinearNormalizer":
    """Load a saved ``LinearNormalizer.state_dict()`` (``spec_normalizer.py`` output)."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    nz = LinearNormalizer()
    nz.load_state_dict(sd)
    meta = sd.get("_spec_meta") if isinstance(sd, dict) else None
    if meta:
        print(f"[normalizer] loaded {path}: {meta.get('summary', meta)}", flush=True)
    return nz


def get_image_range_normalizer() -> SingleFieldLinearNormalizer:
    """Fixed [0,1] -> [-1,1] image normalizer."""
    stat = {
        "min": np.array([0], dtype=np.float32),
        "max": np.array([1], dtype=np.float32),
        "mean": np.array([0.5], dtype=np.float32),
        "std": np.array([np.sqrt(1 / 12)], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(
        np.array([2], dtype=np.float32), np.array([-1], dtype=np.float32), stat
    )
