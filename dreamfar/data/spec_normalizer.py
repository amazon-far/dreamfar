"""Build the SPEC-DERIVED LinearNormalizer (bounds from the YAM URDF + FK, not from a data sample).

Fields: ``action`` = 20-D EE pose (xyz from the FK workspace box + margin; rot6d = exact identity on
[-1,1]; gripper [0,1] + margin), ``state`` = 14-D joints (URDF limits unioned with the vendor spec
+ per-joint margin; gripper [0,1] + margin). Nothing is clipped.

    python -m dreamfar.data.spec_normalizer --urdf yam.urdf --out normalizer.pt
"""
from __future__ import annotations

import argparse

import numpy as np

from .ee_fk import FkChain, load_urdf
from .normalizer import LinearNormalizer, get_bounds_normalizer

# URDF revolute limits (both arms identical) UNIONED with the i2rt vendor spec (degrees). The union takes
# the wider side where they disagree (j1/j2/j3 upper, j4 lower) because the data crosses the tighter one.
URDF_JOINT_LIM = [(-2.61799, 3.13), (0.0, 3.65), (0.0, 3.13),
                  (-1.5708, 1.5708), (-1.5708, 1.5708), (-2.0944, 2.0944)]
VENDOR_JOINT_LIM_DEG = [(-150.0, 180.0), (0.0, 210.0), (0.0, 180.0),
                        (-97.0, 90.0), (-90.0, 90.0), (-120.0, 120.0)]

# Per-joint margins (rad): ~1.7x the worst follower excursion past its soft limit observed per joint,
# floored at 0.15. Grip: the commanded gripper overshoots [0,1] by up to ~0.1. EE: metres past the FK box.
JOINT_MARGIN = [0.15, 0.15, 0.45, 0.50, 0.15, 0.15]
GRIP_MARGIN = 0.20
EE_MARGIN = 0.05
ROT6D_BOUND = 1.0     # |R_ij| <= 1 for an orthonormal column -- exact, not measured


def joint_limits(urdf_path: str) -> np.ndarray:
    """(6, 2) per-arm joint limits: elementwise union of the URDF revolute limits and the vendor spec.
    Asserts the URDF still matches ``URDF_JOINT_LIM`` so a changed URDF fails loudly."""
    urdf = load_urdf(urdf_path)
    jm = {j.name: j for j in urdf.joints}
    per_arm = {s: np.array([[jm[f"{s}_joint{k}"].limit.lower, jm[f"{s}_joint{k}"].limit.upper]
                            for k in range(1, 7)], dtype=np.float64) for s in ("left", "right")}
    if not np.allclose(per_arm["left"], per_arm["right"]):
        raise RuntimeError(f"URDF left/right limits differ:\n{per_arm['left']}\n{per_arm['right']}")
    urdf_lim = per_arm["left"]
    if not np.allclose(urdf_lim, np.array(URDF_JOINT_LIM), atol=1e-5):
        raise RuntimeError(f"URDF at {urdf_path} disagrees with URDF_JOINT_LIM recorded here:\n{urdf_lim}")
    vendor = np.deg2rad(np.array(VENDOR_JOINT_LIM_DEG, dtype=np.float64))
    return np.stack([np.minimum(urdf_lim[:, 0], vendor[:, 0]),
                     np.maximum(urdf_lim[:, 1], vendor[:, 1])], axis=1)


def workspace_box(urdf_path: str, lim: np.ndarray, n: int = 1 << 20, seed: int = 0):
    """Axis-aligned bbox of the reachable flange ORIGIN over the joint-limit box, per arm, in the
    base-relative frame ``ee_fk`` stores: scrambled-Sobol sample through ``FkChain`` + an L-BFGS-B
    polish from the 4 best samples per objective. Returns ``(lo(3), hi(3), diag)``."""
    from scipy.optimize import minimize
    from scipy.stats import qmc
    urdf = load_urdf(urdf_path)
    lo_q, hi_q = lim[:, 0], lim[:, 1]
    bounds = [tuple(r) for r in lim]
    Q = lo_q + qmc.Sobol(d=6, scramble=True, seed=seed).random(n) * (hi_q - lo_q)
    out = {}
    for side in ("left", "right"):
        ch = FkChain(urdf, f"{side}-arm-base", f"{side}-flange-frame",
                     [f"{side}_joint{k}" for k in range(1, 7)])
        P = np.concatenate([ch.flange_poses(Q[i:i + 200_000])[:, :3, 3]
                            for i in range(0, n, 200_000)], axis=0)
        grid = np.stack([P.min(0), P.max(0)])
        pol = grid.copy()
        for axis in range(3):
            order = np.argsort(P[:, axis])
            for row, sign, starts in ((0, +1.0, order[:4]), (1, -1.0, order[-4:])):
                for i in starts:
                    def obj(q, ax=axis, sg=sign, c=ch):
                        return sg * c.flange_poses(q.reshape(1, -1))[0, ax, 3]

                    r = minimize(obj, Q[i], method="L-BFGS-B", bounds=bounds,
                                 options=dict(maxiter=500, ftol=1e-14, gtol=1e-12))
                    v = sign * r.fun
                    pol[row, axis] = min(pol[row, axis], v) if row == 0 else max(pol[row, axis], v)
        out[side] = (pol, np.abs(pol - grid))
    (pl, gl), (pr, gr) = out["left"], out["right"]
    # The two arm chains are identical, so the boxes coincide; the union guards an asymmetric URDF.
    diag = dict(n=n, left=pl.tolist(), right=pr.tolist(),
                grid_polish_gap_max=float(max(gl.max(), gr.max())),
                arms_identical=bool(np.allclose(pl, pr)))
    return np.minimum(pl[0], pr[0]), np.maximum(pl[1], pr[1]), diag


def _joint14_bounds(lim: np.ndarray, margin, grip_margin: float):
    """[L_j1..j6, L_grip, R_j1..j6, R_grip] known bounds. ``margin`` is scalar or per-joint (6,)."""
    m = np.broadcast_to(np.asarray(margin, dtype=np.float64), (6,))
    j_lo, j_hi = lim[:, 0] - m, lim[:, 1] + m
    lo = np.concatenate([j_lo, [-grip_margin], j_lo, [-grip_margin]])
    hi = np.concatenate([j_hi, [1 + grip_margin], j_hi, [1 + grip_margin]])
    return lo, hi


def _ee20_bounds(box_lo, box_hi, ee_margin: float, grip_margin: float):
    """[L_pos(3), L_rot6d(6), L_grip, R_pos(3), R_rot6d(6), R_grip] known bounds."""
    half_lo = np.concatenate([box_lo - ee_margin, np.full(6, -ROT6D_BOUND), [-grip_margin]])
    half_hi = np.concatenate([box_hi + ee_margin, np.full(6, ROT6D_BOUND), [1 + grip_margin]])
    return np.concatenate([half_lo, half_lo]), np.concatenate([half_hi, half_hi])


def build(urdf_path: str, *, n: int = 1 << 20, joint_margin=JOINT_MARGIN,
          grip_margin: float = GRIP_MARGIN, ee_margin: float = EE_MARGIN):
    """Return ``(LinearNormalizer, meta)`` with fields action=ee20, state=joint14."""
    lim = joint_limits(urdf_path)
    box_lo, box_hi, box_diag = workspace_box(urdf_path, lim, n=n)
    nz = LinearNormalizer()
    fields = {}
    for field, kind in (("action", "ee20"), ("state", "joint14")):
        if kind == "joint14":
            lo, hi = _joint14_bounds(lim, joint_margin, grip_margin)
        else:
            lo, hi = _ee20_bounds(box_lo, box_hi, ee_margin, grip_margin)
        nz[field] = get_bounds_normalizer(lo, hi)
        fields[field] = dict(kind=kind, dim=len(lo), min=lo.tolist(), max=hi.tolist())
    meta = dict(
        source="dreamfar.data.spec_normalizer", urdf=urdf_path,
        joint_limits_union=lim.tolist(),
        workspace_box=dict(min=box_lo.tolist(), max=box_hi.tolist(), **box_diag),
        margins=dict(joint=list(np.broadcast_to(np.asarray(joint_margin, dtype=float), (6,))),
                     grip=grip_margin, ee=ee_margin),
        fields={k: v["kind"] for k, v in fields.items()},
        summary=("spec bounds from " + urdf_path + ": "
                 + ", ".join(f"{k}={v['kind']}({v['dim']}D)" for k, v in fields.items())
                 + f"; margins joint {list(np.round(np.broadcast_to(np.asarray(joint_margin, dtype=float), (6,)), 3))} rad "
                 + f"/ grip {grip_margin} / ee {ee_margin} m; 12 rot6d dims are EXACT identity; "
                   "nothing clipped"),
        field_bounds=fields,
    )
    return nz, meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", required=True, help="YAM URDF path")
    p.add_argument("--out", required=True, help="output .pt (torch.save of LinearNormalizer.state_dict())")
    a = p.parse_args()

    import torch

    nz, meta = build(a.urdf)
    wb = meta["workspace_box"]
    print(f"[urdf] joint limits (URDF-vendor union, rad):\n"
          f"{np.round(np.array(meta['joint_limits_union']), 5)}")
    print(f"[fk]   workspace box min={np.round(wb['min'], 4).tolist()} "
          f"max={np.round(wb['max'], 4).tolist()} (n={wb['n']}, grid->polish gap "
          f"{wb['grid_polish_gap_max']:.5f} m, arms identical={wb['arms_identical']})")
    for field, fb in meta["field_bounds"].items():
        lo, hi = np.array(fb["min"]), np.array(fb["max"])
        n_id = int(np.sum((lo == -1.0) & (hi == 1.0)))
        print(f"[{field}] {fb['kind']} {fb['dim']}-D  bound width min={float((hi - lo).min()):.4f} "
              f"max={float((hi - lo).max()):.4f}  exact-identity dims={n_id}")
    sd = nz.state_dict()
    sd["_spec_meta"] = meta
    torch.save(sd, a.out)
    print(f"[saved] {a.out}\n[meta] {meta['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
