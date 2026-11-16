"""Continuous 6D rotation representation (Zhou et al. 2019): the first two COLUMNS of R, flattened
row-major over the (3, 2) block, i.e. d6 = [R00,R01, R10,R11, R20,R21]."""
from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrices -> (..., 6)."""
    R = np.asarray(R)
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def rot6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt; the output is a valid SO(3) element."""
    d6 = np.asarray(d6, dtype=np.float64)
    a1 = d6[..., 0:6:2]                       # column 0 = elements 0,2,4
    a2 = d6[..., 1:6:2]                        # column 1 = elements 1,3,5
    b1 = _normalize(a1)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(a2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)    # columns b1,b2,b3 -> (...,3,3)
