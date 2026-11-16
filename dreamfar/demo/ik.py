"""Differential / iterative IK helpers for the demo's keyboard and mouse admission (numpy only).

`fk` arguments are callables q6 (6,) -> 4x4 base-relative flange pose (one arm's FkChain).
"""
from __future__ import annotations

import math

import numpy as np


def so3_exp(rv: np.ndarray) -> np.ndarray:
    """Rotation vector (3,) -> rotation matrix (3,3) (Rodrigues)."""
    rv = np.asarray(rv, dtype=np.float64)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (3,), robust near 0 and near pi."""
    R = np.asarray(R, dtype=np.float64)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * v
    if math.pi - th < 1e-5:                     # near pi: the antisymmetric part vanishes, use the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0.0, 1.0))
        i = int(np.argmax(k))
        for j in range(3):
            if j != i and A[i, j] < 0:
                k[j] = -k[j]
        return th * k / (np.linalg.norm(k) + 1e-12)
    return (th / (2.0 * math.sin(th))) * v


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (SVD), det +1."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return U @ D @ Vt


def clip_norm(v: np.ndarray, cap: float) -> np.ndarray:
    """Scale `v` down to norm `cap` if longer (direction preserved)."""
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v if n <= cap or n < 1e-12 else v * (cap / n)


def clamp_box(p: np.ndarray, lo, hi) -> np.ndarray:
    """Per-axis clamp of a base-frame position into [lo, hi]."""
    return np.minimum(np.maximum(np.asarray(p, dtype=np.float64), np.asarray(lo, dtype=np.float64)),
                      np.asarray(hi, dtype=np.float64))


def tcp_from_flange(T_flange: np.ndarray, d_tcp) -> tuple:
    """TCP pose from the flange pose; d_tcp = tool offset in the FLANGE frame (m). Returns (p_tcp, R)."""
    T = np.asarray(T_flange, dtype=np.float64)
    R, p = T[:3, :3], T[:3, 3]
    return p + R @ np.asarray(d_tcp, dtype=np.float64), R


def flange_target_from_tcp(p_tcp_t, R_t, d_tcp) -> np.ndarray:
    """The flange position that puts the TCP at `p_tcp_t` with orientation `R_t`."""
    return np.asarray(p_tcp_t, dtype=np.float64) - np.asarray(R_t, dtype=np.float64) @ np.asarray(d_tcp, dtype=np.float64)


def numerical_jacobian(fk, q6: np.ndarray, eps: float = 1e-5) -> tuple:
    """6x6 geometric Jacobian of `fk` (world translation rows, BODY-frame rotation rows) + the pose at q6."""
    q6 = np.asarray(q6, dtype=np.float64)
    T0 = np.asarray(fk(q6), dtype=np.float64)
    p0, R0 = T0[:3, 3], T0[:3, :3]
    J = np.zeros((6, 6))
    for i in range(6):
        qp = q6.copy()
        qp[i] += eps
        T1 = np.asarray(fk(qp), dtype=np.float64)
        J[0:3, i] = (T1[:3, 3] - p0) / eps
        J[3:6, i] = so3_log(R0.T @ T1[:3, :3]) / eps
    return J, T0


def ik_step(fk, q6: np.ndarray, dp: np.ndarray, dw: np.ndarray, damp: float) -> np.ndarray:
    """Keyboard mode: the joint delta that best realises an EE delta (`dp` world translation, `dw`
    body-frame rotation) by damped least squares on the numerical Jacobian. Reachability is enforced
    downstream (joint clamp + FK(q) as the action), not here."""
    J, _ = numerical_jacobian(fk, q6)
    dx = np.concatenate([dp, dw])
    return J.T @ np.linalg.solve(J @ J.T + (damp ** 2) * np.eye(6), dx)


def servo_delta(cur_p, cur_R, tgt_p, tgt_R, max_dp: float, max_dw: float) -> tuple:
    """One bounded step toward the target: world translation `dp` (|dp| <= max_dp) and BODY-frame
    rotation `dw` (|dw| <= max_dw)."""
    dp = clip_norm(np.asarray(tgt_p, dtype=np.float64) - np.asarray(cur_p, dtype=np.float64), max_dp)
    dw = clip_norm(so3_log(np.asarray(cur_R, dtype=np.float64).T @ np.asarray(tgt_R, dtype=np.float64)), max_dw)
    return dp, dw


def ik_servo(fk, q6, p_t, R_t, jlim, q_ref=None, max_dp: float = 0.03, max_dw: float = 0.10, dq_max: float = 0.15,
             damp: float = 0.05, w_pos: float = 50.0, w_rot: float = 1.0, w_post: float = 0.02, w_smooth: float = 0.02,
             iters: int = 6, tol_p: float = 1e-4, tol_w: float = 1e-3, limit_guard: bool = True, dev_tol: float = 0.001,
             dev_frac: float = 1.0, rot_tol: float = 0.05) -> tuple:
    """Regularised iterative differential IK for one 10 Hz slot (mouse mode).

    The target is first bounded relative to the slot's start pose (|dp| <= max_dp, |dw| <= max_dw), then each
    iteration solves the weighted damped least squares
        (J^T W J + (damp^2 + w_smooth + w_post) I) dq = J^T W e + w_post (q_ref - q)
    with an active set on joints parked at a limit, a uniform per-joint velocity cap (dq_max on the slot's total
    motion) and the joint clamp `jlim` (never snapping a joint that starts outside the box). The reach-limit
    guard (`limit_guard`) shortens the step (alpha 1, .6, .3, .1, 0) when the realised TCP motion leaves the
    commanded direction or the orientation error grows, so the arm stops at its reachable boundary instead of
    diving or tilting. Returns (q_new (6,), info) with err_p/err_w to the UNCAPPED target, iters, dq_capped,
    j_clamped, limited (alpha <= 0.3), alpha."""
    q0 = np.asarray(q6, dtype=np.float64).copy()
    q = q0.copy()
    jl = np.asarray([j[0] for j in jlim], dtype=np.float64)
    jh = np.asarray([j[1] for j in jlim], dtype=np.float64)
    q_ref = (jl + jh) / 2.0 if q_ref is None else np.asarray(q_ref, dtype=np.float64)
    jl, jh = np.minimum(jl, q0), np.maximum(jh, q0)
    W = np.diag([w_pos] * 3 + [w_rot] * 3)
    reg = (damp ** 2 + w_smooth + w_post) * np.eye(6)
    T0 = np.asarray(fk(q0), dtype=np.float64)
    dp0, dw0 = servo_delta(T0[:3, 3], T0[:3, :3], p_t, R_t, max_dp, max_dw)
    p_cap = T0[:3, 3] + dp0                              # the slot's bounded target (world translation)
    R_cap = orthonormalize(T0[:3, :3] @ so3_exp(dw0))    # body-frame rotation step from the start pose
    n_it, capped, clamped = 0, False, []
    for n_it in range(1, max(1, int(iters)) + 1):
        J, T = numerical_jacobian(fk, q)
        dp, dw = servo_delta(T[:3, 3], T[:3, :3], p_cap, R_cap, 1e9, 1e9)
        e = np.concatenate([dp, dw])
        if n_it > 1 and float(np.linalg.norm(dp)) < tol_p and float(np.linalg.norm(dw)) < tol_w:
            break                                        # (the first iteration always runs: it carries the posture pull)
        A = J.T @ W @ J + reg
        b = J.T @ W @ e + w_post * (q_ref - q)
        dq = np.linalg.solve(A, b)
        # active set: a joint on a limit that the step would push further out is frozen and the step re-solved
        fz = ((q <= jl + 1e-9) & (dq < 0)) | ((q >= jh - 1e-9) & (dq > 0))
        if fz.any():
            Jf = J.copy()
            Jf[:, fz] = 0.0
            bf = Jf.T @ W @ e + w_post * (q_ref - q)
            bf[fz] = 0.0
            dq = np.linalg.solve(Jf.T @ W @ Jf + reg, bf)
            dq[fz] = 0.0
        q = q + dq
        tot = q - q0                                     # velocity cap on the slot's TOTAL motion, scaled uniformly
        m = float(np.max(np.abs(tot)))
        if m > dq_max:
            capped = True
            q = q0 + tot * (dq_max / m)
        hit = (q <= jl + 1e-9) | (q >= jh - 1e-9)
        q = np.minimum(np.maximum(q, jl), jh)
        clamped = [int(i) for i in np.nonzero(hit)[0]]
        if capped:
            break                                        # the budget is spent; further iterations cannot move
    alpha = 1.0
    if limit_guard:
        c = p_cap - T0[:3, 3]
        cn = float(np.linalg.norm(c))
        ch = c / cn if cn > 1e-6 else None
        e_w0 = float(np.linalg.norm(so3_log(T0[:3, :3].T @ R_cap)))
        q_full = q.copy()
        for alpha in (1.0, 0.6, 0.3, 0.1, 0.0):
            qa = q0 + alpha * (q_full - q0)
            Ta = np.asarray(fk(qa), dtype=np.float64)
            r = Ta[:3, 3] - T0[:3, 3]
            a_ = float(r @ ch) if ch is not None else 0.0
            o_ = float(np.linalg.norm(r - a_ * ch)) if ch is not None else float(np.linalg.norm(r))
            ew = float(np.linalg.norm(so3_log(Ta[:3, :3].T @ R_cap)))
            if o_ <= dev_tol + dev_frac * max(a_, 0.0) and ew <= e_w0 + rot_tol:
                break
        q = qa
    T = np.asarray(fk(q), dtype=np.float64)
    dp, dw = servo_delta(T[:3, 3], T[:3, :3], p_t, R_t, 1e9, 1e9)
    info = {"err_p": float(np.linalg.norm(dp)), "err_w": float(np.linalg.norm(dw)), "iters": n_it,
            "dq_capped": capped, "j_clamped": clamped, "limited": bool(alpha <= 0.3), "alpha": float(alpha)}
    return q, info


def reach_limit_flag(boxed: bool, ik: dict, err_thr: float = 0.02) -> bool:
    """REACH LIMIT warning: the target was clamped by the EE box, or the reach guard cut the step AND the
    arm really lags its target (> err_thr m)."""
    return bool(boxed) or (bool(ik.get("limited")) and float(ik.get("err_p", 0.0)) > float(err_thr))
