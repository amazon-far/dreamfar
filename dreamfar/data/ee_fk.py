"""Forward kinematics: 14-D YAM joint state/action -> 20-D BASE-RELATIVE end-effector pose.

Input  states_actions.bin    (N, 28): [:, :14]=state [:, 14:]=action; each half = [L_j1..6, L_grip, R_j1..6, R_grip].
Output states_actions_ee.bin (N, 40) float32: each 20-half = [L_pos(3), L_rot6d(6), L_grip, R_pos(3), R_rot6d(6), R_grip].
"""
from __future__ import annotations

import json
import os

import numpy as np

from .rot6d import matrix_to_rot6d

# urdfpy 0.0.22 references removed numpy aliases
for _a, _t in [("float", float), ("int", int), ("bool", bool)]:
    if _a not in vars(np):  # hasattr() warns on numpy 1.26
        setattr(np, _a, _t)


def _axis_angle_matrix(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rodrigues: (M,) angles about a FIXED unit `axis` (3,) -> (M,3,3)."""
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = ax
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    c = np.cos(theta)[:, None, None]
    s = np.sin(theta)[:, None, None]
    I = np.eye(3)[None]
    return I + s * K[None] + (1 - c) * (K @ K)[None]


class FkChain:
    """Base-relative FK for ONE arm: the fixed base->flange joint chain, applied to a whole trajectory
    as batched matmuls. Flange(theta) = prod_i [ T_i @ Rot(axis_i, theta_i) ] (fixed joints: Rot=I)."""

    def __init__(self, urdf, base_link: str, ee_link: str, joint_names: list[str]):
        self.joint_names = joint_names
        self._steps = []          # list of (origin4x4, axis|None, jointname|None)
        chain = []
        link = ee_link
        parent_of = {}
        for j in urdf.joints:
            parent_of[j.child] = j
        while link != base_link:
            j = parent_of.get(link)
            if j is None:
                raise RuntimeError(f"no path from {ee_link} to {base_link} at {link}")
            chain.append(j)
            link = j.parent
        chain.reverse()           # base -> ee order
        for j in chain:
            origin = np.asarray(j.origin, dtype=np.float64) if j.origin is not None else np.eye(4)
            if j.joint_type in ("revolute", "continuous"):
                self._steps.append((origin, np.asarray(j.axis, dtype=np.float64), j.name))
            else:
                self._steps.append((origin, None, None))

    def flange_poses(self, joints: np.ndarray) -> np.ndarray:
        """joints: (M, 6) in joint_names order -> (M, 4, 4) base-relative flange transforms."""
        M = joints.shape[0]
        T = np.tile(np.eye(4), (M, 1, 1))
        jidx = {n: k for k, n in enumerate(self.joint_names)}
        for origin, axis, jname in self._steps:
            T = T @ origin[None]
            if axis is not None:
                theta = joints[:, jidx[jname]]
                R = _axis_angle_matrix(axis, theta)
                step = np.tile(np.eye(4), (M, 1, 1))
                step[:, :3, :3] = R
                T = T @ step
        return T


def load_urdf(path: str):
    from urdfpy import URDF
    return URDF.load(path)


def _arm_chains(urdf) -> tuple:
    L = FkChain(urdf, "left-arm-base", "left-flange-frame", [f"left_joint{k}" for k in range(1, 7)])
    R = FkChain(urdf, "right-arm-base", "right-flange-frame", [f"right_joint{k}" for k in range(1, 7)])
    return L, R


class EeFk:
    """Cached joint -> base-relative EE FK for runtime use: ``EeFk(urdf_path)(joint14) -> (N, 20)``."""

    def __init__(self, urdf_path: str):
        self.L, self.R = _arm_chains(load_urdf(urdf_path))

    def __call__(self, joint14: np.ndarray) -> np.ndarray:
        return _half_to_ee(np.asarray(joint14, dtype=np.float64).reshape(-1, 14), self.L, self.R)


def _half_to_ee(half14: np.ndarray, Lchain: "FkChain", Rchain: "FkChain") -> np.ndarray:
    """One 14-D joint half -> 20-D [L_pos(3), L_rot6d(6), L_grip, R_pos(3), R_rot6d(6), R_grip]."""
    Lj, Lg = half14[:, 0:6], half14[:, 6:7]
    Rj, Rg = half14[:, 7:13], half14[:, 13:14]
    Lt = Lchain.flange_poses(Lj)                      # (N,4,4) base-relative
    Rt = Rchain.flange_poses(Rj)
    Lpos, Lrot = Lt[:, :3, 3], matrix_to_rot6d(Lt[:, :3, :3])
    Rpos, Rrot = Rt[:, :3, 3], matrix_to_rot6d(Rt[:, :3, :3])
    return np.concatenate([Lpos, Lrot, Lg, Rpos, Rrot, Rg], axis=1)      # (N,20)


def precompute_ee(state_action_14: np.ndarray, urdf) -> np.ndarray:
    """(N, 28) joint state+action -> (N, 40) base-relative EE state+action."""
    sa = np.asarray(state_action_14, dtype=np.float64).reshape(-1, 28)
    Lchain, Rchain = _arm_chains(urdf)
    state = _half_to_ee(sa[:, :14], Lchain, Rchain)
    action = _half_to_ee(sa[:, 14:], Lchain, Rchain)
    return np.concatenate([state, action], axis=1).astype(np.float32)   # (N,40)


def write_ee_bin(shard_dir: str, urdf, out_name: str = "states_actions_ee.bin") -> int:
    """Write ``<shard_dir>/states_actions_ee.bin`` (N, 40) float32 next to the shard's joint bin,
    episode by episode per ``index.json``. Returns the frame count."""
    idx = json.load(open(os.path.join(shard_dir, "index.json")))
    sa_cols = int(idx.get("sa_cols", 28))
    mm = np.memmap(os.path.join(shard_dir, "states_actions.bin"), dtype=np.float32, mode="r").reshape(-1, sa_cols)
    out = np.zeros((mm.shape[0], 40), dtype=np.float32)
    for e in idx["episodes"]:
        s, n = int(e["start"]), int(e["size"])
        out[s:s + n] = precompute_ee(mm[s:s + n], urdf)
    out.tofile(os.path.join(shard_dir, out_name))
    return int(mm.shape[0])
