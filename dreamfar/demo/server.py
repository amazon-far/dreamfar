"""Interactive demo: drive the streaming DreamFAR world model from the browser with the keyboard or the mouse.

DEMAND-DRIVEN. The model does nothing until an input arrives. `DynamicsModule.rollout` owns its own loop but
admits a frame only when `pending_fn()` says an action is queued, lets `on_admit(frame)` write that frame's
action row, fires `on_commit(frame, latent)` the instant a frame commits, and blocks in `wait_fn()` when the
pyramid has drained and the queue is empty (KV cache frozen, zero GPU work). Two decoupled queues: the server
action queue (the 10 Hz ticker appends one ABSOLUTE target per tick while a key is held / a mouse target is
away from the arm, independent of the model's commit rate) and the client's frame buffer (steady playout).
Actions are absolute 20-D EE poses [L_pos(3), L_rot6d(6), L_grip, R_pos(3), R_rot6d(6), R_grip] = FK of a
joint vector that is clamped to the data box, so every command is reachable by construction. Producers hand
RAW rows to on_admit, the one place they are normalized (model.cond_action).
Run with `uvicorn dreamfar.demo.server:app` (dreamfar/demo/README.md); the env is read at import.
"""
from __future__ import annotations

import asyncio
import collections
import copy
import json
import logging
import math
import os
import queue
import struct
import threading
import time
from collections import Counter

import cv2
import numpy as np
import torch
from einops import rearrange
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from dreamfar.data.ee_fk import EeFk
from dreamfar.inference.build import build_model, decode_frame, decode_noise
from dreamfar.inference.episodes import list_episodes, load_episode
from dreamfar.inference.seeding import build_regime, committed_window

from .ik import (clamp_box, flange_target_from_tcp, ik_servo, ik_step, reach_limit_flag, so3_exp, so3_log,
                 tcp_from_flange)
from .modes import ECTRL_SPEC, MCTRL_SPEC, MODES, key_names, mode_transition, normalize_mode

HERE = os.path.dirname(os.path.abspath(__file__))


def _req(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} must be set (see dreamfar/demo/README.md)")
    return v


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "off")


# ---- required inputs ----------------------------------------------------------------------------------------
CKPT = _req("DEMO_CKPT")                      # dynamics checkpoint (.ckpt)
CONFIG = _req("DEMO_CONFIG")                  # its config (.hydra/config.yaml or a bare algorithm node)
LATENT_ROOT = _req("LATENT_ROOT")             # precomputed latent shards (seed scenes)
URDF_PATH = _req("URDF_PATH")                 # YAM URDF (forward kinematics)
HELDOUT_PATH = os.environ.get("HELDOUT_PATH") or None    # {"asset_ids": [...]}: the scenes offered (else every episode)
CONTACT_DIR = os.environ.get("CONTACT_DIR") or None      # per-arm contact labels for the mid-start keyframe bank
# Inference settings applied on top of the config (the paper's streaming setting by default).
OVERRIDES = {"dyn_flow_steps": int(os.environ.get("DYN_FLOW_STEPS", "16")),
             "infer_t_schedule": os.environ.get("INFER_T_SCHEDULE", "logit_normal"),
             "stream_ctx_noise": float(os.environ.get("STREAM_CTX_NOISE", "0.2404252730845196")),
             "infer_weights_dtype": os.environ.get("INFER_WEIGHTS_DTYPE", "bf16"),
             "ee_urdf_path": URDF_PATH}
# Tokenizer files, if the config's paths do not apply on this machine (model-zoo configs hold placeholders).
for _env, _key in (("TOKENIZER_CKPT", "tokenizer_ckpt"), ("TOKENIZER_CONFIG", "tokenizer_config")):
    if os.environ.get(_env):
        OVERRIDES[_key] = os.environ[_env]

# ---- seeding ------------------------------------------------------------------------------------------------
# dup = duplicated cold start (the committed window is filled with copies of GT frame 0); mid = the first
# SEED_FRAMES GT frames (10 Hz) are seeded first, for "take over mid-task". The page switches it per episode.
SEED_MODE = os.environ.get("DEMO_SEED_MODE", "mid")
SEED_FRAMES = int(os.environ.get("DEMO_SEED_FRAMES", "100"))
EP = int(os.environ.get("DEMO_EP", "-1"))     # -1 => random scene; else an index into the scene list
# Optional task filter: `;`-separated case-insensitive PREFIXES on the catalog task_name.
TASK_FILTER = [t.strip().lower() for t in os.environ.get("DEMO_TASKS", "").split(";") if t.strip()]
TASK_LIST_MAX = int(os.environ.get("DEMO_TASK_LIST_MAX", "400")) or None
# Rollout horizon: how many frames one generation can ever COMMIT (not a time limit; idle is free). Costs
# 2 x total x one latent of GPU memory; on exhaustion the server reseeds and says so.
TOTAL = int(os.environ.get("DEMO_TOTAL", "8000"))
DECODE_STEPS = int(os.environ.get("DEMO_DECODE_STEPS", "2"))

# ---- keyboard (eefik) ---------------------------------------------------------------------------------------
# Steps per committed frame: gripper 0.1 (the data's median transition rate), translation 0.01 m (ordinary
# teleop speed, 10 cm/s at 10 Hz), rotation 0.03 rad (body frame). IK_DAMP = damped-least-squares lambda.
GRIP_STEP = float(os.environ.get("DEMO_GRIP_STEP", "0.1"))
EE_POS_STEP = float(os.environ.get("DEMO_EE_POS_STEP", "0.01"))
EE_ROT_STEP = float(os.environ.get("DEMO_EE_ROT_STEP", "0.03"))
IK_DAMP = float(os.environ.get("DEMO_IK_DAMP", "0.05"))
# Initial input mode of a new session (keyboard | mouse); the page switches it live ({"type":"mode"}).
_cm = os.environ.get("DEMO_CTRL_MODE", "keyboard")
CTRL_MODE = normalize_mode(_cm)
if CTRL_MODE is None:
    raise SystemExit(f"DEMO_CTRL_MODE must be keyboard|mouse (got {_cm!r})")
MODE: dict = {"cur": CTRL_MODE}

# ---- GPU / compile ------------------------------------------------------------------------------------------
# DEMO_PIPELINE: auto = decode on a second visible GPU when there is one (fps = 1/max(predict, decode)
# instead of 1/(predict + decode)); 0 forces serial; 1 requests it. The decoder is torch.compiled and warmed
# at startup so the first interactive frame does not pay the compile.
PIPELINE = os.environ.get("DEMO_PIPELINE", "auto")
COMPILE_DECODER = _flag("DEMO_COMPILE_DECODER", "1")
WARM_DYNAMICS = _flag("DEMO_WARM_DYNAMICS", "1")
RECOMPILE_LIMIT = int(os.environ.get("DEMO_RECOMPILE_LIMIT", "64"))

# ---- iterative IK (mouse admission) -------------------------------------------------------------------------
# TCP = jaw centre, DEMO_HAND_TCP metres from the flange in the flange frame (+z = approach). The absolute EE
# box is the arm's reach inside the data joint box (FK envelope over 20k samples), inset by the margin.
_tcp = os.environ.get("DEMO_HAND_TCP", "0,0,0.12").split(",")
HAND_TCP = np.array([float(x) for x in _tcp], dtype=np.float64) if len(_tcp) == 3 else np.array([0.0, 0.0, 0.12])
HAND_EE_BOX_MARGIN = float(os.environ.get("DEMO_HAND_EE_BOX_MARGIN", "0.0"))
HAND_IK_ITERS = int(os.environ.get("DEMO_HAND_IK_ITERS", "6"))
HAND_IK_DAMP = float(os.environ.get("DEMO_HAND_IK_DAMP", "0.05"))
HAND_REACH_GUARD = _flag("DEMO_HAND_REACH_GUARD", "1")
HAND_MAX_DQ = float(os.environ.get("DEMO_HAND_MAX_DQ", "0.15"))     # rad per slot (data p99 per-frame |dq| 0.12-0.18)
HAND_W_POS = float(os.environ.get("DEMO_HAND_W_POS", "50"))
HAND_W_ROT = float(os.environ.get("DEMO_HAND_W_ROT", "1"))
HAND_W_POST = float(os.environ.get("DEMO_HAND_W_POST", "0.02"))
HAND_W_SMOOTH = float(os.environ.get("DEMO_HAND_W_SMOOTH", "0.02"))

# ---- mouse --------------------------------------------------------------------------------------------------
# Left / right click on the TOP view selects the LEFT / RIGHT arm; the pointer maps ABSOLUTELY to that arm's
# TCP x,y inside its table-plane box (stretched over the view along MOUSE_AXES, no camera unprojection); the
# wheel moves z; MCTRL keys turn it (world frame, about the TCP); Space / middle click toggles its gripper.
# Slots are filled at admission by ik_servo; the pointer target is EMA-smoothed and the servo rate-limited.
MOUSE_MAX_DP = float(os.environ.get("DEMO_MOUSE_MAX_DP", "0.04"))     # m per committed frame
MOUSE_MAX_DW = float(os.environ.get("DEMO_MOUSE_MAX_DW", "0.10"))     # rad per committed frame
MOUSE_ROT_STEP = float(os.environ.get("DEMO_MOUSE_ROT_STEP", "0.05"))  # rad per 10 Hz tick while a key is held
MOUSE_ROT_LEAD = float(os.environ.get("DEMO_MOUSE_ROT_LEAD", "0.30"))  # max rad the rotation target runs ahead of the arm
MOUSE_DZ = float(os.environ.get("DEMO_MOUSE_DZ", "0.01"))             # m per wheel notch (100 px of deltaY)
MOUSE_SMOOTH = float(os.environ.get("DEMO_MOUSE_SMOOTH", "0.5"))      # per-tick EMA of the pointer target (1 = none)
MOUSE_DGRIP = float(os.environ.get("DEMO_MOUSE_DGRIP", "0.15"))       # gripper slew per frame
MOUSE_EPS_P = float(os.environ.get("DEMO_MOUSE_EPS_P", "0.003"))      # m: closer than this = at target (hold)
MOUSE_EPS_W = float(os.environ.get("DEMO_MOUSE_EPS_W", "0.01"))       # rad
MOUSE_MAX_AQ = int(os.environ.get("DEMO_MOUSE_MAX_AQ", "2"))          # queued-slot cap: no backlog, short pointer-to-pixel
# AXES: the base-frame axis along the image's RIGHT and the one along its DOWN ("<sign><axis>,<sign><axis>").
# The top camera looks forward over the bases (image bottom): image right = -y, image down = -x.
_max = os.environ.get("DEMO_MOUSE_AXES", "-y,-x").replace(" ", "").split(",")
MOUSE_AXES = tuple((("xy".index(a[1])), (1.0 if a[0] == "+" else -1.0)) for a in _max)
if len(MOUSE_AXES) != 2 or {a for a, _ in MOUSE_AXES} != {0, 1}:
    raise SystemExit(f"DEMO_MOUSE_AXES must be like '-y,+x' (got {_max!r})")
# Per-arm TCP box (arm base frame, m) the view is stretched over: the data p1..p99 padded 3 cm, z floored at
# the table. DEMO_MOUSE_BOX_L / _R = "x0,x1,y0,y1,z0,z1" override one arm.
_MBOX = {"L": "0.080,0.590,-0.490,0.120,0.000,0.540", "R": "0.080,0.600,-0.110,0.520,0.000,0.550"}
MOUSE_BOX = {}
for _a in ("L", "R"):
    _b = [float(x) for x in os.environ.get(f"DEMO_MOUSE_BOX_{_a}", _MBOX[_a]).split(",")]
    MOUSE_BOX[_a] = (np.array(_b[0::2]), np.array(_b[1::2]))
# Arm (re)selection: a click that CHANGES the arm starts a MOUSE_GRACE_S hold (the page draws a countdown
# ring); when it ends the pointer's position THEN becomes the target, so the arm never jumps to where the
# pointer was steering the other arm.
MOUSE_GRACE_S = float(os.environ.get("DEMO_MOUSE_GRACE_S", "1.0"))
# Spawn at a good grasping pose (mouse mode, mid-start only): the handover cut is chosen so the selected arm's
# flange rotation is closest to the data's table-grasp mode, its gripper open and its TCP above ZMIN.
MOUSE_SPAWN = _flag("DEMO_MOUSE_SPAWN", "1")
MOUSE_GRASP_R6 = {"L": [-0.0113, -0.9095, -0.8982, -0.1734, 0.4394, -0.3778],
                  "R": [0.1906, -0.9157, -0.9614, -0.1012, -0.1986, -0.3888]}
MOUSE_SPAWN_ZMIN = float(os.environ.get("DEMO_MOUSE_SPAWN_ZMIN", "0.10"))
MOUSE_SPAWN_OPEN = float(os.environ.get("DEMO_MOUSE_SPAWN_OPEN", "0.8"))
MOUSE_SPAWN_MARGIN = int(os.environ.get("DEMO_MOUSE_SPAWN_MARGIN", "100"))   # frames kept after the cut
MOUSE_SPAWN_VPEN = float(os.environ.get("DEMO_MOUSE_SPAWN_VPEN", "200"))    # deg per m/frame speed penalty

# ---- streaming ----------------------------------------------------------------------------------------------
ACTION_HZ = float(os.environ.get("DEMO_ACTION_HZ", "10"))
QUEUE_MAX = int(os.environ.get("DEMO_QUEUE_MAX", "600"))
JPEG_Q = int(os.environ.get("DEMO_JPEG_Q", "85"))
# Display order of the views (display only; the latent layout is the model's obs_keys order).
VIEW_ORDER = [v.strip() for v in os.environ.get("DEMO_VIEW_ORDER", "left,top,right").split(",")]

L_GRIP, R_GRIP = 9, 19
# Frame header: t_input(f64) frame(i32) gL gR(f32) action-queue depth(i32) L_pos(3f) R_pos(3f) in-flight(i32);
# the JPEG panel follows (index.html reads it from offset HDR.size = 52).
HDR = struct.Struct(">diffi6fi")
KILL = "kill"        # gate sentinel
SLOT = "slot"        # gate payload of a mouse slot: filled AT ADMISSION (mouse_admit) from the newest target
LIVE = object()      # mouse_step() -> "some arm is away from its target: queue a slot"

# Joint clamp = the data's per-joint p0.5 / p99.5 (keeps commands in-distribution), gripper [0, 1].
# [L_j1..L_j6, L_grip, R_j1..R_j6, R_grip]
JLIM = ([(-1.346, 0.419), (0.213, 2.534), (0.098, 2.448), (-1.427, 1.350), (-0.870, 1.212), (-1.674, 1.224), (0.0, 1.0)] +
        [(-0.382, 1.393), (0.283, 2.583), (0.137, 2.513), (-1.423, 1.320), (-1.217, 0.865), (-1.204, 1.594), (0.0, 1.0)])
Q_REF = [0.5 * (lo + hi) for lo, hi in JLIM]           # posture reference of the iterative IK
EE_BOX: dict = {}                                        # arm -> (lo3, hi3) base-frame flange box (filled in model())
# Stepped key tables: (up, dn, arm, kind, axis, step, label).
ECTRL = [(u, d, a, k, ax, (EE_POS_STEP if st == "p" else EE_ROT_STEP if st == "o" else GRIP_STEP), lb)
         for u, d, a, k, ax, st, lb in ECTRL_SPEC]
MCTRL = [(u, d, a, k, ax, MOUSE_ROT_STEP, lb) for u, d, a, k, ax, _st, lb in MCTRL_SPEC]
_CTRL_TABLES = {"eefik": ECTRL, "mouse": MCTRL}


def ctrl_of(mode: str) -> list:
    return _CTRL_TABLES[mode]


def _r6_to_mat(d6) -> np.ndarray:
    a1, a2 = np.asarray(d6[0:6:2], dtype=np.float64), np.asarray(d6[1:6:2], dtype=np.float64)
    b1 = a1 / np.linalg.norm(a1)
    a2 = a2 - float(b1 @ a2) * b1
    b2 = a2 / np.linalg.norm(a2)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


MOUSE_GRASP_R = {a: _r6_to_mat(v) for a, v in MOUSE_GRASP_R6.items()}


# ---- scene selection (one starting scene per task; a new scene is a new task) -------------------------------
def norm_task(t: str | None) -> str:
    return " ".join((t or "").split()).lower()


def one_per_task(items, task_of, rank=None) -> list:
    """First item per task in `rank` order (default: list order); survivors keep their list order."""
    keys = [(rank(it) if rank else 0, i) for i, it in enumerate(items)]
    keep, seen = set(), set()
    for _, i in sorted(keys):
        k = norm_task(task_of(items[i]))
        if k not in seen:
            seen.add(k)
            keep.add(i)
    return [it for i, it in enumerate(items) if i in keep]


def next_pool(tasks, cands, want, history, first: bool) -> list:
    """Indices (among `cands`) to draw the next scene from: an explicit pick (the first request, or a task other
    than the one on screen) -> that task (exact name, else prefix); otherwise a task not shown yet this session,
    and once all were shown any task but the current one."""
    cands = list(cands)
    cur = norm_task(history[-1]) if history else ""
    w = norm_task(want)
    if w and (first or w != cur):
        exact = [i for i in cands if norm_task(tasks[i]) == w]
        sub = exact or [i for i in cands if norm_task(tasks[i]).startswith(w)]
        if sub:
            return sub
    seen = {norm_task(t) for t in history}
    fresh = [i for i in cands if norm_task(tasks[i]) not in seen]
    if fresh:
        return fresh
    other = [i for i in cands if norm_task(tasks[i]) != cur]
    return other or cands


# ---- mouse geometry -----------------------------------------------------------------------------------------
def mouse_uv_to_xy(arm: str, u: float, v: float) -> np.ndarray:
    """Normalised TOP-view pointer (u right, v down, in [0,1]) -> the arm's TCP (x, y) in its base frame."""
    lo, hi = MOUSE_BOX[arm]
    out = np.zeros(2)
    for (ax, sg), w in zip(MOUSE_AXES, (u, v)):
        f = min(1.0, max(0.0, float(w)))
        out[ax] = lo[ax] + (f if sg > 0 else 1.0 - f) * (hi[ax] - lo[ax])
    return out


def mouse_xy_to_uv(arm: str, xy) -> list:
    """Inverse of mouse_uv_to_xy (unclamped: a pose outside the box maps outside [0,1])."""
    lo, hi = MOUSE_BOX[arm]
    uv = [0.0, 0.0]
    for k, (ax, sg) in enumerate(MOUSE_AXES):
        f = (float(xy[ax]) - lo[ax]) / max(1e-9, hi[ax] - lo[ax])
        uv[k] = f if sg > 0 else 1.0 - f
    return uv


def mouse_spawn_cut(M: dict, q14: np.ndarray, a20: np.ndarray, arm_pref: str | None, n_seed: int) -> tuple[int, dict]:
    """The mid-start handover frame c for mouse mode (see MOUSE_SPAWN): q14 = (T,14) GT joint state, a20 = (T,20)
    RAW commanded EE action, both at 10 Hz. Scores the held pose (frame c-1) for every c in [n_seed, T - margin]."""
    T = int(q14.shape[0])
    c_lo, c_hi = n_seed, T - MOUSE_SPAWN_MARGIN
    info = {"cut": n_seed, "fallback": True, "T": T, "arm": arm_pref, "n_ok": 0, "lo": c_lo, "hi": c_hi}
    if c_hi < c_lo:
        return n_seed, info
    k = np.arange(c_lo, c_hi + 1) - 1                               # the held frame of every candidate cut
    best = None
    for arm, off, chain in (("L", 0, M["fk"].L), ("R", 7, M["fk"].R)):
        if arm_pref in ("L", "R") and arm != arm_pref:
            continue
        Tf = chain.flange_poses(np.asarray(q14[:, off:off + 6], dtype=np.float64))
        R = Tf[:, :3, :3]
        tcp = Tf[:, :3, 3] + R @ HAND_TCP
        ang = np.degrees(np.arccos(np.clip((np.einsum("tij,ij->t", R, MOUSE_GRASP_R[arm]) - 1.0) / 2.0, -1.0, 1.0)))
        spd = np.r_[0.0, np.linalg.norm(np.diff(tcp, axis=0), axis=1)]
        g_st, g_cmd = q14[:, off + 6], a20[:, 9 if arm == "L" else 19]
        ok = (g_st[k] > MOUSE_SPAWN_OPEN) & (g_cmd[k] > MOUSE_SPAWN_OPEN) & (tcp[k, 2] > MOUSE_SPAWN_ZMIN)
        sc = ang[k] + MOUSE_SPAWN_VPEN * spd[k]
        info[f"n_ok_{arm}"] = int(ok.sum())
        info["n_ok"] += int(ok.sum())
        if ok.any():
            j = int(np.argmin(np.where(ok, sc, np.inf)))
            if best is None or sc[j] < best[0]:
                best = (float(sc[j]), arm, int(k[j]) + 1, float(ang[k[j]]), float(tcp[k[j], 2]), float(g_st[k[j]]), float(spd[k[j]]))
        info[f"angle_at_default_{arm}"] = round(float(ang[n_seed - 1]), 1)
    if best is None:
        return n_seed, info
    info.update(cut=best[2], fallback=False, arm=best[1], score=round(best[0], 1), angle_deg=round(best[3], 1),
                tcp_z=round(best[4], 3), grip=round(best[5], 3), speed_m_per_frame=round(best[6], 4))
    return best[2], info


# ---- process state ------------------------------------------------------------------------------------------
app = FastAPI()
_M: dict = {}
_BUILD_LOCK = threading.Lock()
_GPU = threading.Lock()             # one rollout at a time
_LIVE: dict = {"sess": None}        # the session holding _GPU, so a new one can EVICT it


class _Abort(Exception):
    """Raised from on_commit / on_admit to unwind the rollout (reset / evict / disconnect)."""


def _gate_put(sess_: dict, tok) -> None:
    """Enqueue one gate token and wake a rollout idling in wait()."""
    sess_["gate"].put(tok)
    k = sess_.get("kick")
    if k is not None:
        k.set()


def _gate_kill_now(sess_: dict) -> None:
    """KILL at the FRONT of the gate (Reset / New episode / mode switch): the rollout unwinds at its very next
    admission instead of behind every queued action. Consumers only get_nowait() under qsize() > 0."""
    g = sess_["gate"]
    with g.mutex:
        g.queue.appendleft(KILL)
        g.not_empty.notify()
    k = sess_.get("kick")
    if k is not None:
        k.set()


_CUDAGRAPH_MSGS: list = []          # inductor warnings mentioning cudagraphs during the decoder compile


class _CgHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "cudagraph" in msg.lower():
            _CUDAGRAPH_MSGS.append(msg[:200])


def compile_status(net, tok=None) -> dict:
    """What is actually compiled in this process."""
    import torch._dynamo.eval_frame as _ef
    dyn_dec = getattr(net, "_cmp_stream_decode", None) is not None
    dec = tok.decoder if tok is not None else None
    dec_c = isinstance(dec, _ef.OptimizedModule) if dec is not None else False
    wdt = str(next(net.dynamics.parameters()).dtype).replace("torch.", "")
    return {"dynamics_stream_decode": dyn_dec, "dynamics_snapshot": getattr(net, "_cmp_stream_snap", None) is not None,
            "dynamics_weights_dtype": {"bfloat16": "bf16", "float32": "fp32"}.get(wdt, wdt),
            "dynamics_mode": "default" if dyn_dec else None,
            "decoder": dec_c, "decoder_mode": ("reduce-overhead" if dec_c else None),
            "decoder_cudagraphs": ("skipped: " + _CUDAGRAPH_MSGS[0]) if _CUDAGRAPH_MSGS else ("on" if dec_c else "off"),
            "dynamics_compiled_at_load": (_M.get("dyn_compiled_at_load") or {}).get("ms") if _M else None}


def _dec_tok(M: dict):
    """The tokenizer that decodes for the viewer (the decode GPU's copy when pipelined)."""
    return (M["pipe"] or {}).get("tok", M["model"].tokenizer)


def _compile_and_warm(tok, dev: str, net, noise, where: str) -> None:
    """torch.compile the decoder (reduce-overhead = CUDA graphs) and run it once so the compile is paid at
    startup. Static shapes -> one graph; an eager pass first populates lazily-built caches outside capture."""
    logging.getLogger("torch._inductor").addHandler(_CgHandler())
    logging.getLogger("torch._inductor.cudagraph_trees").addHandler(_CgHandler())
    n_views = len(net.obs_keys)
    h = int(net.image_size) // int(net.tokenizer.encoder.patch_size)
    zc = int(net.z_dim)
    import torch._dynamo as _dyn
    _dyn.config.recompile_limit = RECOMPILE_LIMIT     # the decoder specialises per (dtype, euler step); 8 is too few
    with torch.no_grad():
        for dt in (torch.bfloat16, torch.float32):
            decode_frame(tok, torch.zeros((1, zc, n_views * h, h), device=dev, dtype=dt), n_views, noise, DECODE_STEPS)
    torch.cuda.synchronize(dev)
    tok.decoder = torch.compile(tok.decoder, mode="reduce-overhead", dynamic=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        for dt in (torch.bfloat16, torch.float32):      # committed latents arrive in the model's working dtype
            dummy = torch.zeros((1, zc, n_views * h, h), device=dev, dtype=dt)
            for _ in range(2):
                decode_frame(tok, dummy, n_views, noise, DECODE_STEPS)
    torch.cuda.synchronize(dev)
    print(f"[demo] decoder torch.compile()d + warmed on the {where} ({time.perf_counter() - t0:.1f}s, "
          f"latent {tuple(dummy.shape)}, bf16+fp32, recompile_limit={_dyn.config.recompile_limit})", flush=True)


def _ensure_ee_box(M: dict) -> None:
    """Absolute base-frame EE box per arm = the flange's reach inside the data joint box, inset by the margin."""
    rng = np.random.default_rng(0)
    for arm, off_, ch in (("L", 0, M["fk"].L), ("R", 7, M["fk"].R)):
        lo_j = np.array([JLIM[off_ + i][0] for i in range(6)])
        hi_j = np.array([JLIM[off_ + i][1] for i in range(6)])
        P_ = ch.flange_poses(lo_j + rng.random((20000, 6)) * (hi_j - lo_j))[:, :3, 3]
        EE_BOX[arm] = (P_.min(axis=0) + HAND_EE_BOX_MARGIN, P_.max(axis=0) - HAND_EE_BOX_MARGIN)
    print("[demo] EE box: " + "; ".join(f"{a}=[{np.round(b[0], 3).tolist()} .. {np.round(b[1], 3).tolist()}]"
                                         for a, b in EE_BOX.items()), flush=True)


def model() -> dict:
    with _BUILD_LOCK:
        if not _M:
            dev = "cuda:0"
            net = build_model(CKPT, CONFIG, device=dev, ema=True, overrides=OVERRIDES)
            try:
                step = int(torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True).get("global_step") or 0)
            except Exception:  # noqa: BLE001
                step = 0
            val, meta = list_episodes(LATENT_ROOT, heldout_path=HELDOUT_PATH)
            if TASK_FILTER:
                keep = [s for s in val if (s.task_name or "").strip().lower().startswith(tuple(TASK_FILTER))]
                if not keep:
                    raise SystemExit(f"DEMO_TASKS={TASK_FILTER} matched 0 of {len(val)} episodes")
                print(f"[demo] DEMO_TASKS: {len(keep)}/{len(val)} episodes", flush=True)
                val = keep
            # ONE starting scene per task, in the held-out list's order (else the catalog order).
            try:
                rank = {a: i for i, a in enumerate(json.load(open(HELDOUT_PATH)).get("asset_ids") or [])} if HELDOUT_PATH else {}
            except Exception:  # noqa: BLE001
                rank = {}
            n_all = len(val)
            val = one_per_task(val, lambda s: s.task_name, rank=lambda s: rank.get(s.asset_id, len(rank)))
            print(f"[demo] scenes: one per task -> {len(val)} of {n_all} episodes", flush=True)
            if not val:
                raise SystemExit(f"no episodes under {LATENT_ROOT}" + (f" in {HELDOUT_PATH}" if HELDOUT_PATH else ""))
            res = int(net.image_size)
            ist = net.normalizer["action"].params_dict["input_stats"]
            _M.update(model=net, val=val, meta=meta, dev=dev, noise=decode_noise(net, dev),
                      lo=ist["min"].float().cpu().numpy(), hi=ist["max"].float().cpu().numpy(),
                      fk=EeFk(URDF_PATH),
                      tasks=Counter((s.task_name or "").strip() for s in val).most_common(),
                      model_id=dict(ckpt=CKPT, step=step, weights_dtype=compile_status(net)["dynamics_weights_dtype"],
                                    res=res, z_dim=int(net.z_dim), P=int(net.dyn_flow_steps),
                                    ctx_noise=str(net.stream_ctx_noise), t_schedule=str(net.infer_t_schedule),
                                    n_tokens=int(net.n_tokens), n_memory=int(net.n_memory), seed_mode=SEED_MODE,
                                    n_dup=int(committed_window(net))))
            _ensure_ee_box(_M)
            # ---- optional SECOND GPU for the decoder -------------------------------------------------------
            n_gpu = torch.cuda.device_count()
            want = (n_gpu > 1) if PIPELINE == "auto" else (PIPELINE not in ("0", "off", "false"))
            _M["pipe"] = None
            if want and n_gpu > 1:
                dev_b = "cuda:1"
                tok_b = copy.deepcopy(net.tokenizer).to(dev_b).eval()    # a copy of the verified frozen tokenizer
                _M["pipe"] = dict(tok=tok_b, dev=dev_b, noise=decode_noise(net, dev_b))  # same seed -> same noise
                if COMPILE_DECODER:
                    _compile_and_warm(tok_b, dev_b, net, _M["pipe"]["noise"], "decode GPU")
                print(f"[demo] PIPELINED: predict on cuda:0, decode on {dev_b} ({n_gpu} GPUs visible)", flush=True)
            elif want:
                print(f"[demo] pipeline requested but only {n_gpu} GPU visible: running SERIAL", flush=True)
            if _M["pipe"] is None and COMPILE_DECODER:
                _compile_and_warm(net.tokenizer, dev, net, _M["noise"], "predict GPU")
            print(f"[demo] views={net.obs_keys} P={net.dyn_flow_steps} ctx_noise={net.stream_ctx_noise} "
                  f"t_schedule={net.infer_t_schedule} n_tokens={net.n_tokens} total={TOTAL} "
                  f"grip L=[{_M['lo'][L_GRIP]:.4f},{_M['hi'][L_GRIP]:.4f}] R=[{_M['lo'][R_GRIP]:.4f},{_M['hi'][R_GRIP]:.4f}]",
                  flush=True)
    return _M


def seed_episode(ep: int | None = None, task: str | None = None, seed_mode: str | None = None,
                 spawn: str | None = None, history: tuple | list = (), first: bool = True) -> dict:
    """Rollout kwargs for one generation over a TOTAL-length action horizon that HOLDS the seed pose.

    dup: GT frame 0 duplicated over the committed window; mid: SEED_FRAMES real frames seeded first. `task` is a
    prefix on the catalog task_name, `ep` pins a scene index; `history` / `first` drive next_pool. `spawn`
    ("L" / "R" / "*", mouse mode, mid only) searches the handover cut for a good grasping pose."""
    M = model()
    net, val, dev = M["model"], M["val"], M["dev"]
    seed_mode = seed_mode or SEED_MODE
    dup = seed_mode == "dup"
    n_seed = 1 if dup else SEED_FRAMES
    t_seed0 = time.perf_counter()
    # dup only uses frame 0, but build_regime("dup") clamps n_dup to the loaded length: load a window's worth.
    need = (committed_window(net) + 1) if dup else (n_seed + 1)
    spawn = spawn if (spawn and not dup and MOUSE_SPAWN) else None
    if ep is None:
        cands = [i for i, s in enumerate(val) if (s.size - 1) // 3 + 1 >= need]
        if task and not any(norm_task(val[i].task_name).startswith(norm_task(task)) for i in cands):
            print(f"[demo] task {task!r} matched 0 episodes; drawing a new task", flush=True)
        pool = next_pool([s.task_name for s in val], cands, task, list(history), first)
        ep = EP if EP >= 0 else int(np.random.default_rng().choice(pool))
    aid = val[ep].asset_id
    batch = load_episode(net, val[ep], M["meta"], dev, cap=(10 ** 7 if spawn else need), contact_dir=CONTACT_DIR)
    z_gt = rearrange(net.stack_views(batch["latent"]), "t b c h w -> b t c h w")
    act_ep = net.cond_action(batch["action"])
    st_ep, fi_ep = batch["state"], batch["frame_idx"]
    contact = batch["contact"] if "contact" in batch else None
    cut, cut_info = n_seed, None
    if spawn:
        t_sp = time.perf_counter()
        a20 = net.normalizer["action"].unnormalize(act_ep[0].float()).cpu().numpy()
        cut, cut_info = mouse_spawn_cut(M, st_ep[0, :, :14].float().cpu().numpy(), a20, spawn if spawn in ("L", "R") else None, n_seed)
        cut_info["search_ms"] = round((time.perf_counter() - t_sp) * 1000.0, 1)
        sl = slice(cut - n_seed, cut + 1)                # the seed window + 1 frame, as `need`
        z_gt, act_ep, st_ep, fi_ep = z_gt[:, sl], act_ep[:, sl], st_ep[:, sl], fi_ep[:, sl]
        if contact is not None:
            contact = contact[:, sl]
        print(f"[demo] MOUSE SPAWN ep={ep} cut={cut} ({cut_info})", flush=True)
    ns = max(1, min(z_gt.shape[1] - 1, n_seed))         # what build_regime("mid") will pick
    if dup:
        # Hold frame 0's own action/state for the whole horizon: re-feeding the command that produced the
        # frame is "stay where you are" in this absolute action space.
        action = act_ep[:, 0:1].repeat(1, TOTAL, 1).contiguous()
        state = st_ep[:, 0:1].repeat(1, TOTAL, 1).contiguous()
        hold_row = 0
    else:
        action = act_ep[:, ns - 1:ns].repeat(1, TOTAL, 1).contiguous()
        action[:, :ns] = act_ep[:, :ns]                  # the seeded region keeps its own actions
        state = st_ep[:, ns - 1:ns].repeat(1, TOTAL, 1).contiguous()
        state[:, :ns] = st_ep[:, :ns]
        hold_row = ns - 1
    step = int(fi_ep[0, 1] - fi_ep[0, 0]) if fi_ep.shape[1] > 1 else 3
    frame_idx = (fi_ep[0, 0] + step * torch.arange(TOTAL, device=dev))[None]
    kw = build_regime("dup" if dup else "mid", net, z_gt, action, state, frame_idx,
                      contact=contact, seed_sec=n_seed / 10.0, fps=10)
    load_ms = (time.perf_counter() - t_seed0) * 1000.0
    print(f"[demo] SEED ep={ep} aid={aid} task={val[ep].task_name!r} mode={seed_mode} n_seed={kw['n_seed']} "
          f"pad={kw['pad']} total={TOTAL} contact={'yes' if contact is not None else 'no'} load {load_ms:.0f} ms", flush=True)
    return {"kw": kw, "ep": ep, "aid": aid, "task": val[ep].task_name,
            "held_pose": net.normalizer["action"].unnormalize(action[0, hold_row].float()).cpu().numpy().copy(),
            "seed_cut": (cut if not dup else None), "seed_cut_info": cut_info,
            "seed_mode": seed_mode, "load_ms": load_ms}


def _to_u8(rgb: torch.Tensor) -> np.ndarray:
    """(V,3,H,W) float [0,1] -> (V,H,W,3) uint8 on the host (one device->host copy per frame)."""
    return rgb.mul(255).round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _view_order(net) -> list:
    keys_ = list(net.obs_keys)
    order = [keys_.index(v) for v in VIEW_ORDER if v in keys_]
    return order + [i for i in range(len(keys_)) if i not in order]


def _panel_jpeg(u8: np.ndarray) -> bytes:
    """(V,H,W,3) uint8 -> one [view | view | view] JPEG in VIEW_ORDER (pure CPU)."""
    panel = np.concatenate([u8[i] for i in _view_order(model()["model"])], axis=1)
    ok, buf = cv2.imencode(".jpg", panel[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
    return buf.tobytes() if ok else b""


def _warm_dynamics() -> None:
    """torch.compile the dynamics at load: `rollout(compiled=True)` builds its compiled decode/snapshot lazily on
    the first call and caches them on the module, so pay it here on a throw-away dup rollout of P+2 frames. The
    rollout seeds its own generator per call, so this cannot perturb a later session."""
    M = model()
    if not WARM_DYNAMICS:
        return
    net = M["model"]
    t0 = time.perf_counter()
    n = int(net.dyn_flow_steps) + 2
    S = seed_episode(ep=None, seed_mode="dup")
    kw = S["kw"]
    left = [n]

    def on_admit(fi: int) -> None:
        left[0] -= 1                       # the action tensor already holds the seed pose for every row

    with torch.no_grad():
        net.rollout(kw["z_hist"], kw["action"], kw["state"], kw["frame_idx"],
                    seed_latents=kw["seed_latents"], seed_contact=kw["seed_contact"],
                    compiled=True, on_commit=lambda i, z: None,
                    pending_fn=lambda: left[0], on_admit=on_admit, wait_fn=lambda: False)
    torch.cuda.synchronize(M["dev"])
    ms = (time.perf_counter() - t0) * 1000.0
    M["dyn_compiled_at_load"] = {"ms": round(ms), "frames": n, "ep": S["ep"]}
    print(f"[demo] dynamics torch.compile()d + warmed at load ({ms / 1000:.1f}s, {n} frames of ep {S['ep']}); "
          f"compiled={compile_status(net, _dec_tok(M))}", flush=True)


@app.on_event("startup")
def _warm() -> None:
    """Build the model BEFORE serving (it blocks the event loop for minutes), then compile + warm the dynamics."""
    model()
    _warm_dynamics()


def _catalog():
    if not _M:
        return None
    h = _LIVE.get("hello") or {}
    return {"modes": list(MODES), "tasks": (_M.get("tasks") or [])[:TASK_LIST_MAX],
            "ctrl_mode": MODE["cur"], "task": h.get("task") or ""}


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(open(os.path.join(HERE, "index.html")).read(), headers={"Cache-Control": "no-store"})


@app.get("/catalog")
def catalog_json():
    """Modes + tasks (no session needed)."""
    return JSONResponse({"catalog": _catalog()}, headers={"Cache-Control": "no-store"})


@app.get("/state")
def state_json():
    """The live session's hello + state, for probes that must not open a second /ws (which would evict the viewer)."""
    return JSONResponse({"hello": _LIVE.get("hello"), "session": _LIVE["sess"] is not None,
                         "ctrl_mode": MODE["cur"], "mouse": _LIVE.get("mouse_state"), "state": _LIVE.get("state"),
                         "model": (_M.get("model_id") if _M else None), "ctx_hist": _LIVE.get("ctx_hist"),
                         "compiled": (compile_status(_M["model"], _dec_tok(_M)) if _M else None),
                         "dyn_compiled_at_load": (_M.get("dyn_compiled_at_load") if _M else None)})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    M = model()
    net = M["model"]
    if MODE["cur"] != CTRL_MODE:
        MODE["cur"] = CTRL_MODE          # the input mode is a SESSION property: a new viewer starts from the default
    # ---- EVICT any previous session instead of queueing behind it: its rollout is either blocked in the gate
    #      (unblocked by KILL) or between commits (aborts within one commit).
    prev = _LIVE["sess"]
    if prev is not None:
        prev["alive"] = False
        _gate_put(prev, KILL)
        print("[demo] evicting the previous session", flush=True)
    await loop.run_in_executor(None, _GPU.acquire)
    sess = {"alive": True, "gate": queue.Queue(), "kick": threading.Event()}
    _LIVE["sess"] = sess
    keys = {k: 0 for k in key_names(ctrl_of(MODE["cur"]))}   # rebuilt from the session's mode at every (re)seed
    # OUTBOUND: control JSON through a FIFO that is never dropped; video through a LATEST-WINS slot of depth 1.
    outj: collections.deque = collections.deque()
    slot: list = [None]
    wake = asyncio.Event()
    box = {"t_client": 0.0, "reset": None, "state": "starting", "committed": 0,
           "q": [0.0] * 14, "enqueued": 0, "dropped": 0, "inflight": 0,
           "task": None, "mem": 0, "bank": 0, "S": None, "seed_mode": SEED_MODE,
           "dyn_ms": [], "dec_ms": [], "mem_frames": [], "prologue_ms": None, "ws_drop": 0,
           "mode": MODE["cur"], "ctrl": ctrl_of(MODE["cur"]),   # this GENERATION's input mode + key table
           "ctx_hist": {},                                      # committed-KV flow-time level -> #frames
           # MOUSE: the selected arm + `active` survive reseeds; tgt / R / g = per-arm TCP target, rotation target,
           # gripper target (set from the seed pose at every generation start); raw = the pointer's mapped x,y
           # (None = off the view: hold); ver counts input events, stall_ver = the ver at which the servo stalled;
           # grace_until = deadline of the selection hold, pend = the newest mapped x,y seen during it.
           "mouse": {"arm": None, "active": False, "tgt": {}, "R": {}, "g": {}, "raw": {"L": None, "R": None}, "ver": 0,
                     "stall_ver": -1, "n_still": 0, "t_src": 0.0, "taps": {}, "n_msg": 0, "last_push": None,
                     "grace_until": None, "pend": None, "ik": {}}}

    def emit(b: bytes) -> None:
        """Binary frame: latest wins; a frame still waiting is replaced, not queued."""
        if slot[0] is not None:
            box["ws_drop"] += 1
            if box["ws_drop"] in (1, 10, 100) or box["ws_drop"] % 1000 == 0:
                print(f"[demo] viewer behind: {box['ws_drop']} frame(s) replaced before send (latest-wins)", flush=True)
        slot[0] = b
        wake.set()

    def push(obj: dict) -> None:
        """Control JSON: FIFO, never dropped."""
        outj.append(json.dumps(obj))
        wake.set()

    def mode_switch(want: str) -> None:
        """Live input-mode switch, part 1: every key is released; part 2 is the caller's reset, after which the
        next generation snapshots MODE["cur"] and sends the new hello. Nothing is reloaded."""
        prev_mode = MODE["cur"]
        MODE["cur"] = want
        for k_ in keys:
            keys[k_] = 0
        print(f"[demo] mode {prev_mode} -> {want}: reseeding", flush=True)

    def mouse_input(m: dict) -> None:
        """One {"type":"mouse"} message: `arm` "L"/"R" = a click selecting that arm, `u`,`v` in [0,1] = the pointer
        over the TOP view with `in` false once it left (= hold), `dz` = wheel notches (+ = up), `grip` "toggle".
        Only TARGETS change here; the ticker smooths + paces them and mouse_admit servos the arm."""
        ms = box["mouse"]
        if not ms["tgt"]:
            return                                              # generation not initialised yet (seed prologue)
        ms["n_msg"] += 1
        try:
            ms["t_src"] = float(m.get("t") or 0.0) / 1000.0     # the browser clock (ms) -> frame header
        except (TypeError, ValueError):
            ms["t_src"] = 0.0
        a = m.get("arm")
        picked = a in ("L", "R") and (ms["arm"] != a or not ms["active"])   # a click that CHANGES the arm, or the first
        if a in ("L", "R"):
            if picked:
                print(f"[demo] mouse: arm {ms['arm']} -> {a}", flush=True)
            ms["arm"], ms["active"] = a, True
            ms["ver"] += 1
        arm = ms["arm"]
        if not ms["active"] or arm is None:
            return
        if picked:
            ms["raw"][arm] = None                               # the newly picked arm starts by HOLDING where it is
            ms["grace_until"], ms["pend"] = time.perf_counter() + MOUSE_GRACE_S, None
        if "u" in m or "in" in m:
            try:
                inside = bool(m.get("in", True)) and m.get("u") is not None
                uv = (float(m["u"]), float(m["v"])) if inside else None
            except (TypeError, ValueError, KeyError):
                uv = None
            if uv is not None and not all(math.isfinite(x) for x in uv):
                uv = None
            xy = mouse_uv_to_xy(arm, *uv) if uv is not None else None
            if ms["grace_until"] is not None:
                ms["pend"] = xy                                 # during the hold only remembered; applied at the deadline
            else:
                ms["raw"][arm] = xy
            ms["ver"] += 1
        if m.get("dz") is not None:
            try:
                dz = max(-20.0, min(20.0, float(m["dz"])))
            except (TypeError, ValueError):
                dz = 0.0
            if dz and math.isfinite(dz):
                lo, hi = MOUSE_BOX[arm]
                z = float(ms["tgt"][arm][2])
                ms["tgt"][arm][2] = min(max(hi[2], z), max(min(lo[2], z), z + dz * MOUSE_DZ))   # never snaps an outside z
                ms["ver"] += 1
        if m.get("grip") == "toggle":
            ms["g"][arm] = 0.0 if ms["g"][arm] > 0.5 else 1.0
            print(f"[demo] mouse: {arm} gripper -> {'OPEN' if ms['g'][arm] > 0.5 else 'CLOSED'}", flush=True)
            ms["ver"] += 1

    async def recv_loop() -> None:
        try:
            while True:
                raw_msg = await ws.receive()
                if raw_msg.get("type") == "websocket.disconnect":
                    raise ConnectionError("viewer disconnected")
                if not raw_msg.get("text"):
                    continue
                m = json.loads(raw_msg["text"])
                t = m.get("type")
                if t == "keys":
                    k = m.get("keys", {})
                    taps = box["mouse"]["taps"]
                    for name in keys:
                        nv = 1 if k.get(name) else 0
                        if nv and not keys[name]:
                            taps[name] = 1              # rising edge: a tap shorter than one tick still turns one step
                        keys[name] = nv
                    box["t_client"] = float(m.get("t_client", 0.0))
                elif t == "mouse":
                    if box["mode"] == "mouse":
                        mouse_input(m)
                elif t == "ping":
                    push({"type": "pong", "t": m.get("t"), "t_srv": time.time() * 1000.0})
                elif t == "reset":
                    box["reset"] = "new" if m.get("new_episode") else "same"
                    box["task"] = (m.get("task") or None) if m.get("new_episode") else box["task"]
                    if m.get("seed_mode") in ("dup", "mid"):
                        box["seed_mode"] = m["seed_mode"]        # takes effect from THIS reseed on
                    if m.get("ep") is not None:
                        box["ep_pin"] = int(m["ep"])
                    _gate_kill_now(sess)             # unblock an idle rollout / cut a streaming one at its next admission
                elif t == "mode":
                    want = normalize_mode(m.get("mode"))
                    if want is None:
                        push({"type": "error", "error": "unknown mode"})
                    else:
                        tr = mode_transition(MODE["cur"], want, bool(m.get("new_episode")))
                        if tr != "ack":
                            mode_switch(want)
                        push({"type": "mode", "mode": want, "ok": True, "reset": tr})   # the ack precedes the hello
                        if tr != "ack":
                            box["reset"] = "new" if tr == "reset_new" else "same"
                            if tr == "reset_new":
                                box["task"] = m.get("task") or None
                            _gate_kill_now(sess)
        except Exception:  # noqa: BLE001
            sess["alive"] = False
            _gate_put(sess, KILL)

    async def ticker() -> None:
        """THE INTEGRATOR AND THE PRODUCER of the action queue: one absolute target per 1/ACTION_HZ, at a fixed rate
        independent of the model's commit rate. Keyboard targets are integrated HERE at enqueue time (a drained
        backlog must render exactly what was asked); mouse slots are filled at admission from the newest target."""
        dt = 1.0 / max(1e-6, ACTION_HZ)
        fkm = M["fk"]

        def mouse_step():
            """Per 10 Hz tick: orientation keys -> the selected arm's rotation target (one step per held tick, at
            least one per tap, never more than MOUSE_ROT_LEAD ahead), the pointer's mapped x,y -> an EMA into the
            arm's TCP target; LIVE if any arm is away from its target (else None = hold, the model idles). A servo
            that stopped making progress holds until the next input event. Pushes `mouse_state` when it changed."""
            ms = box["mouse"]
            if not ms["tgt"]:
                return None
            q, arm, taps = box["q"], ms["arm"], ms["taps"]
            poses = {}
            for a_, off, chain in (("L", 0, fkm.L), ("R", 7, fkm.R)):
                Tf = chain.flange_poses(np.asarray(q[off:off + 6], dtype=np.float64)[None, :])[0]
                poses[a_] = tcp_from_flange(Tf, HAND_TCP)
            if ms["active"] and arm in ("L", "R"):
                dw = np.zeros(3)
                for up, dn, _a, _k, ax, stp, _l in MCTRL:
                    dw[ax] += stp * ((1 if (keys.get(up) or taps.get(up)) else 0) - (1 if (keys.get(dn) or taps.get(dn)) else 0))
                if dw.any():
                    R_now = poses[arm][1]
                    Rn = so3_exp(dw) @ ms["R"][arm]
                    lead_n = float(np.linalg.norm(so3_log(R_now.T @ Rn)))
                    if lead_n <= MOUSE_ROT_LEAD or lead_n < float(np.linalg.norm(so3_log(R_now.T @ ms["R"][arm]))):
                        ms["R"][arm] = Rn
                        ms["ver"] += 1
            taps.clear()
            grace_left = 0.0
            if ms["grace_until"] is not None and arm in ("L", "R"):
                grace_left = ms["grace_until"] - time.perf_counter()
                if grace_left <= 0.0:                      # the hold ends: the pointer's newest position becomes the target
                    ms["grace_until"], grace_left = None, 0.0
                    ms["raw"][arm], ms["pend"] = ms["pend"], None
                    ms["ver"] += 1
            for a_ in ("L", "R"):
                raw = ms["raw"].get(a_)
                if raw is not None:
                    t_ = ms["tgt"][a_]
                    d_ = raw - t_[:2]
                    t_[:2] = raw if float(np.max(np.abs(d_))) < 5e-4 else t_[:2] + MOUSE_SMOOTH * d_
            need, err = False, {}
            for a_, off in (("L", 0), ("R", 7)):
                p_, R_ = poses[a_]
                ep_ = float(np.linalg.norm(ms["tgt"][a_] - p_))
                ew_ = float(np.linalg.norm(so3_log(R_.T @ ms["R"][a_])))
                eg_ = abs(float(ms["g"][a_]) - float(q[off + 6]))
                err[a_] = (ep_, ew_, eg_)
                if ep_ > MOUSE_EPS_P or ew_ > MOUSE_EPS_W or eg_ > 0.01:
                    need = True
            stalled = ms["stall_ver"] == ms["ver"]
            st = {"type": "mouse_state", "arm": arm, "active": ms["active"], "moving": bool(need and not stalled),
                  "tgt": {a_: [round(float(x), 4) for x in ms["tgt"][a_]] for a_ in ("L", "R")},
                  "ee": {a_: [round(float(x), 4) for x in poses[a_][0]] for a_ in ("L", "R")},
                  "uv_tgt": {a_: [round(x, 4) for x in mouse_xy_to_uv(a_, ms["tgt"][a_])] for a_ in ("L", "R")},
                  "uv_ee": {a_: [round(x, 4) for x in mouse_xy_to_uv(a_, poses[a_][0])] for a_ in ("L", "R")},
                  "z_box": {a_: [round(float(MOUSE_BOX[a_][0][2]), 3), round(float(MOUSE_BOX[a_][1][2]), 3)] for a_ in ("L", "R")},
                  "g_tgt": {a_: round(float(ms["g"][a_]), 2) for a_ in ("L", "R")},
                  "g": {a_: round(float(q[o_ + 6]), 3) for a_, o_ in (("L", 0), ("R", 7))},
                  "err_mm": {a_: round(err[a_][0] * 1000.0, 1) for a_ in err},
                  "err_deg": {a_: round(math.degrees(err[a_][1]), 1) for a_ in err},
                  "limit": {a_: bool((ms.get("ik") or {}).get(a_, {}).get("limited")) for a_ in ("L", "R")},
                  "grace": round(max(0.0, grace_left), 1),      # s left of the selection hold (0 = tracking)
                  "stalled": stalled, "n_msg": ms["n_msg"]}
            key_ = json.dumps({k_: v_ for k_, v_ in st.items() if k_ != "n_msg"})
            if key_ != ms["last_push"]:
                ms["last_push"] = key_
                _LIVE["mouse_state"] = st
                loop.call_soon_threadsafe(push, st)
            return LIVE if (need and not stalled) else None

        try:
            t_next = loop.time() + dt
            while True:
                # Absolute-deadline pacing: a stall longer than one period resyncs instead of bursting.
                await asyncio.sleep(max(0.0, t_next - loop.time()))
                t_now = loop.time()
                t_next += dt
                if t_now - t_next > dt:
                    t_next = t_now + dt
                mode = box["mode"]                 # this generation's input mode (a switch lands via the reset path)
                if mode == "mouse":
                    if mouse_step() is None:       # every arm at its target: nothing queued, KV cache frozen
                        continue
                    if sess["gate"].qsize() >= MOUSE_MAX_AQ:
                        continue                   # no backlog: every slot reads the newest target at admission
                elif not any(keys.values()):
                    continue
                if sess["gate"].qsize() >= QUEUE_MAX:
                    box["dropped"] += 1
                    continue
                if mode == "mouse":
                    # The tick only PACES (one slot = one model frame); the slot is filled at admission (mouse_admit),
                    # so its frame is conditioned on the newest target instead of one 20-90 ms old.
                    box["enqueued"] += 1
                    _gate_put(sess, (SLOT, None, None))
                else:                              # eefik: keys -> EE delta -> differential IK -> joint clamp -> FK(q)
                    q = box["q"]
                    for arm, off, chain in (("L", 0, fkm.L), ("R", 7, fkm.R)):
                        dp = np.zeros(3); dw = np.zeros(3); dg = 0.0
                        for up, dn, a, kind, ax, stp, _lab in ECTRL:
                            if a != arm:
                                continue
                            d = keys[up] - keys[dn]
                            if not d:
                                continue
                            if kind == "t":   dp[ax] += stp * d
                            elif kind == "r": dw[ax] += stp * d
                            else:             dg     += stp * d
                        if dp.any() or dw.any():
                            dq = ik_step(lambda qq, ch=chain: ch.flange_poses(np.asarray(qq, dtype=np.float64)[None, :])[0],
                                         np.asarray(q[off:off + 6], dtype=np.float64), dp, dw, IK_DAMP)
                            for i in range(6):
                                lo, hi = JLIM[off + i]
                                q[off + i] = min(hi, max(lo, q[off + i] + float(dq[i])))
                        if dg:
                            lo, hi = JLIM[off + 6]
                            q[off + 6] = min(hi, max(lo, q[off + 6] + dg))
                    ee = fkm(np.asarray(q, dtype=np.float64)[None, :])[0]
                    box["enqueued"] += 1
                    _gate_put(sess, (tuple(float(x) for x in ee), box["t_client"] / 1000.0, tuple(q)))
        except asyncio.CancelledError:
            return

    def run_generation(S: dict) -> None:
        """One rollout, gated by sess['gate']. Returns when it aborts or exhausts TOTAL."""
        kw = S["kw"]
        action, n_seed = kw["action"], kw["n_seed"]
        mode = box["mode"]                        # fixed for this generation (set at (re)seed from MODE["cur"])
        # The commanded joint vector starts from the seed JOINT state, so the first FK output continues the
        # recorded pose exactly.
        st14 = kw["state"][0, n_seed].float().cpu().numpy()
        box["q"] = [float(x) for x in st14[:14]]
        if mode == "mouse":
            # Every arm's target = where it is now (nothing moves until the pointer / wheel / keys say so); the arm
            # selection and `active` carry over; before any click the spawn search's arm is the suggestion.
            ms = box["mouse"]
            ms["taps"].clear()
            ms.update(raw={"L": None, "R": None}, stall_ver=-1, n_still=0, ik={}, last_push=None, grace_until=None, pend=None)
            tgt_, R_, g_ = {}, {}, {}
            for a_, off, chain in (("L", 0, M["fk"].L), ("R", 7, M["fk"].R)):
                Tf = chain.flange_poses(np.asarray(box["q"][off:off + 6], dtype=np.float64)[None, :])[0]
                p_, Rm_ = tcp_from_flange(Tf, HAND_TCP)
                tgt_[a_], R_[a_], g_[a_] = p_.copy(), Rm_.copy(), float(box["q"][off + 6])
            ms["R"], ms["g"], ms["tgt"] = R_, g_, tgt_
            if ms["arm"] is None and (S.get("seed_cut_info") or {}).get("arm") in ("L", "R"):
                ms["arm"] = S["seed_cut_info"]["arm"]
            print(f"[demo] mouse generation start: arm {ms['arm']} ({'active' if ms['active'] else 'click to take control'}), "
                  f"TCP L {np.round(tgt_['L'], 3).tolist()} R {np.round(tgt_['R'], 3).tolist()}, seed_cut {S.get('seed_cut')}", flush=True)
        t_stream = [0.0]
        t_gen0 = time.perf_counter()
        n_local = [0]
        af0 = [0]                             # newest admitted frame index, for status messages
        frame_meta: dict = {}                 # frame index -> (input timestamp, commanded q14, raw action row)

        # Pinned host rings for the per-frame values read off the device by the decode worker (never a CUDA sync on
        # the rollout thread); a slot is consumed within a couple of frames. `cs0` is a cuda:0 side stream that
        # waits on the commit event, so the dynamics stream is never blocked on the decode GPU.
        RING = 64
        A_DIM = int(action.shape[2])
        pin_in = torch.empty((RING, A_DIM), dtype=torch.float32).pin_memory()     # H2D staging (on_admit)
        pin_row = torch.empty((RING, A_DIM), dtype=torch.float32).pin_memory()    # D2H: the row the frame saw
        pin_st = torch.empty((RING, 14), dtype=torch.float32).pin_memory()        # D2H: predicted state
        cs0 = torch.cuda.Stream(device=M["dev"])
        pipe = M["pipe"]
        tok_b, dev_b, noise_b = ((pipe["tok"], pipe["dev"], pipe["noise"]) if pipe is not None
                                 else (net.tokenizer, M["dev"], M["noise"]))

        def _ship(u8: np.ndarray, hdr: bytes) -> None:
            """uint8 views -> JPEG -> viewer. Runs on the decode worker (pipelined) or the rollout thread (serial)."""
            loop.call_soon_threadsafe(emit, hdr + _panel_jpeg(u8))

        def _finish(z: torch.Tensor, ev: torch.cuda.Event, info: dict) -> None:
            """Committed latent -> pixels -> JPEG -> viewer. All device reads happen HERE, after `ev`."""
            i = info["i"]
            k = i % RING
            cs0.wait_event(ev)
            with torch.cuda.stream(cs0):
                pin_row[k].copy_(action[0, info["ci"]].float(), non_blocking=True)
                stp = info["st_pred"]
                if stp is not None:
                    pin_st[k].copy_(stp[i, 0].float(), non_blocking=True)
                z_b = z.to(dev_b, non_blocking=True)          # peer copy from the side stream (no-op serial)
            cs0.synchronize()                                  # host waits for the tiny copies, nothing else
            t_d0 = time.perf_counter()
            with torch.no_grad():
                rgb = decode_frame(tok_b, z_b, len(net.obs_keys), noise_b, DECODE_STEPS)
            torch.cuda.synchronize(dev_b)
            box["dec_ms"].append((time.perf_counter() - t_d0) * 1000.0); del box["dec_ms"][:-40]
            raw = info["raw"]
            if raw is None:
                raw = net.normalizer["action"].unnormalize(torch.as_tensor(pin_row[k].numpy().copy(), device=M["dev"])).cpu().numpy()
            _ts = info["t_src"]
            hdr = HDR.pack(_ts * 1000.0 if _ts else box["t_client"], i, float(raw[L_GRIP]), float(raw[R_GRIP]), info["aq"],
                           *[float(raw[d]) for d in (0, 1, 2, 10, 11, 12)], info["inflight"])
            _ship(_to_u8(rgb), hdr)

        def _show_seed_thread() -> None:
            """The viewer sees the scene the model starts from (the LAST seed frame, index n_seed-1) the moment a
            generation starts, instead of a black panel until the first input. Viewer-only: bypasses on_commit."""
            n = int(kw.get("n_seed") or 0)
            if kw.get("seed_latents") is None or n < 1:
                return
            try:
                torch.cuda.set_device(dev_b)
                t0 = time.perf_counter()
                z = kw["seed_latents"][:, n - 1].to(dev_b, dtype=next(net.dynamics.parameters()).dtype)
                with torch.no_grad():
                    rgb = decode_frame(tok_b, z, len(net.obs_keys), noise_b, DECODE_STEPS)
                torch.cuda.synchronize(dev_b)
                st = kw["state"][0, n - 1].float().cpu().numpy().tolist()[:14]
                raw = M["fk"](np.asarray(st, dtype=np.float64)[None, :])[0]   # FK of the GT joint state, like live frames
                hdr = HDR.pack(0.0, n - 1, float(raw[L_GRIP]), float(raw[R_GRIP]), 0,
                               *[float(raw[d]) for d in (0, 1, 2, 10, 11, 12)], 0)
                _ship(_to_u8(rgb), hdr)
                print(f"[demo] seed frame {n - 1} shipped at generation start ({(time.perf_counter() - t0) * 1000:.0f} ms)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[demo] seed frame show FAILED: {e!r}", flush=True)

        threading.Thread(target=_show_seed_thread, daemon=True, name="seed-show").start()

        def on_commit(i: int, z: torch.Tensor) -> None:
            if not sess["alive"]:
                raise _Abort()
            box["committed"] += 1
            box["inflight"] = max(0, box["inflight"] - 1)   # this frame has just LEFT the pyramid
            n_local[0] += 1
            ci = max(0, i - 1)                              # the row that CONDITIONED this frame (act[i-1])
            _ts, _q14, _raw = frame_meta.pop(i, (0.0, None, None))
            # Read the hook per commit: the rollout (re)binds `_stream_dbg` when it starts.
            dbg = getattr(net, "_stream_dbg", None) or {}
            _lv = dbg.get("ctx_lv")
            if _lv is not None:
                _k = f"{float(_lv[min(max(i, 0), len(_lv) - 1)]):g}"
                box["ctx_hist"][_k] = box["ctx_hist"].get(_k, 0) + 1
                _LIVE["ctx_hist"] = box["ctx_hist"]
            msl = dbg.get("mem_sel_log")
            box["mem"] = len(msl[-1]) if msl else 0
            box["mem_frames"] = [int(f) for f in msl[-1]] if msl else []
            box["bank"] = len(dbg.get("mem_pool") or ())
            ta = box.get("t_admit")
            if ta:
                box["dyn_ms"].append((time.perf_counter() - ta) * 1000.0); del box["dyn_ms"][:-40]
            if not t_stream[0]:
                t_stream[0] = time.perf_counter()
            if box["prologue_ms"] is None:
                box["prologue_ms"] = (time.perf_counter() - t_gen0) * 1000.0
                print(f"[demo] seed prologue -> first commit {box['prologue_ms']:.0f} ms; "
                      f"compiled={compile_status(net, _dec_tok(M))}", flush=True)
            info = {"i": i, "ci": ci, "t_src": _ts, "q14": _q14, "raw": _raw, "st_pred": dbg.get("st_pred"),
                    "aq": sess["gate"].qsize(), "inflight": box["inflight"]}
            # ONE event on the dynamics stream: xs[i], st_pred[i] and the action row are final behind it.
            ev = torch.cuda.Event()
            ev.record()
            if pipe is not None:
                dq.put((z, ev, info))
            else:
                _finish(z, ev, info)
            if n_local[0] % 20 == 0 and t_stream[0]:
                el = time.perf_counter() - t_stream[0]
                print(f"[demo] frame {i} n={n_local[0]} {n_local[0] / max(el, 1e-9):.2f} fps "
                      f"gpu_alloc {torch.cuda.memory_allocated() / 2**30:.2f} peak "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
                      f"aq={sess['gate'].qsize()} enq={box['enqueued']} drop={box['dropped']}", flush=True)
            _state("streaming" if (box["inflight"] or sess["gate"].qsize()) else "idle", i)

        def _state(sname: str, i: int) -> None:
            if box["state"] != sname:
                box["state"] = sname
                _LIVE["state"] = sname
                loop.call_soon_threadsafe(push, {"type": "state", "state": sname, "committed": box["committed"], "frame": i,
                                                 "aq": sess["gate"].qsize(), "inflight": box["inflight"],
                                                 "enq": box["enqueued"], "drop": box["dropped"]})

        # ---- ASYNC DECODE WORKER (two-GPU pipeline): on_commit hands (latent, event, info) over and returns at
        #      once; a single FIFO worker keeps frames in commit order.
        dq: queue.Queue = queue.Queue()

        def _decode_loop() -> None:
            torch.cuda.set_device(dev_b)
            while True:
                item = dq.get()
                if item is None:
                    return
                try:
                    _finish(*item)
                except Exception as e:  # noqa: BLE001 -- never kill the session over one frame
                    print(f"[demo] decode worker error: {e!r}", flush=True)

        dth = None
        if pipe is not None:
            dth = threading.Thread(target=_decode_loop, daemon=True)
            dth.start()

        def pending() -> int:
            """Actions queued and NOT yet admitted to the pyramid."""
            return sess["gate"].qsize()

        def wait() -> bool:
            """Called by the rollout only when the pyramid is EMPTY and nothing is pending: true idle, KV cache
            frozen. False => unwind the rollout."""
            _state("idle", af0[0])
            kick = sess["kick"]
            while sess["alive"]:
                if sess["gate"].qsize() > 0:
                    return True
                kick.wait(0.05)                    # set by _gate_put; the timeout only re-checks `alive`
                kick.clear()
            return False

        def mouse_admit():
            """Fill a mouse slot NOW (rollout thread, at admission): every arm away from its target takes ONE bounded
            ik_servo step toward it (TCP -> flange target, clamped to the EE box widened to contain the current pose;
            |dp| <= MOUSE_MAX_DP, |dw| <= MOUSE_MAX_DW, joint-velocity cap, reach guard, joint clamp) and its gripper
            slews; an arm at its target is not touched. Three slots without motion mark the servo stalled (reach
            limit) until the next input event. Then FK -> the 20-D action."""
            q, ms, fkm = box["q"], box["mouse"], M["fk"]
            moved = 0.0
            for arm, off, chain in (("L", 0, fkm.L), ("R", 7, fkm.R)):
                p_t, R_t = ms["tgt"].get(arm), ms["R"].get(arm)
                if p_t is None or R_t is None:
                    continue
                p_t = np.array(p_t, dtype=np.float64)
                q6 = np.asarray(q[off:off + 6], dtype=np.float64)
                fk6 = (lambda qq, ch=chain: ch.flange_poses(np.asarray(qq, dtype=np.float64)[None, :])[0])
                T = fk6(q6)
                p_tcp, R_now = tcp_from_flange(T, HAND_TCP)
                if (float(np.linalg.norm(p_t - p_tcp)) > MOUSE_EPS_P
                        or float(np.linalg.norm(so3_log(R_now.T @ R_t))) > MOUSE_EPS_W):
                    p_fl_t = flange_target_from_tcp(p_t, R_t, HAND_TCP)
                    lo_b, hi_b = EE_BOX[arm]
                    p_box = clamp_box(p_fl_t, np.minimum(lo_b, T[:3, 3]), np.maximum(hi_b, T[:3, 3]))
                    boxed = bool(np.any(np.abs(p_box - p_fl_t) > 1e-9))
                    q6n, ik = ik_servo(fk6, q6, p_box, R_t, JLIM[off:off + 6], q_ref=Q_REF[off:off + 6],
                                       max_dp=MOUSE_MAX_DP, max_dw=MOUSE_MAX_DW, dq_max=HAND_MAX_DQ, damp=HAND_IK_DAMP,
                                       w_pos=HAND_W_POS, w_rot=HAND_W_ROT, w_post=HAND_W_POST, w_smooth=HAND_W_SMOOTH,
                                       iters=HAND_IK_ITERS, limit_guard=HAND_REACH_GUARD)
                    moved = max(moved, float(np.max(np.abs(q6n - q6))))
                    for i in range(6):
                        q[off + i] = float(q6n[i])
                    ms["ik"][arm] = {"limited": reach_limit_flag(boxed, ik), "err_p": round(ik["err_p"], 4)}
                g_t = ms["g"].get(arm)
                if g_t is not None:
                    gl, gh = JLIM[off + 6]
                    dg = min(MOUSE_DGRIP, max(-MOUSE_DGRIP, min(gh, max(gl, float(g_t))) - q[off + 6]))
                    q[off + 6] += dg
                    moved = max(moved, abs(dg))
            if moved < 1e-4:
                ms["n_still"] += 1
                if ms["n_still"] >= 3:
                    ms["stall_ver"] = ms["ver"]
            else:
                ms["n_still"] = 0
            ee = fkm(np.asarray(q, dtype=np.float64)[None, :])[0]
            return tuple(float(x) for x in ee), ms.get("t_src", 0.0), tuple(q)

        def on_admit(fi: int) -> None:
            """One queued action -> the frame just admitted, written ONCE to the row that conditions it (frame f is
            conditioned on act[f-1]). Every producer hands a RAW 20-D row; this is the one place it is normalized."""
            tok = sess["gate"].get_nowait()
            if tok == KILL or not sess["alive"]:
                raise _Abort()
            tok, t_src, q14 = tok
            box["t_admit"] = time.perf_counter()
            if tok is SLOT:
                tok, t_src, q14 = mouse_admit()
            row = max(0, min(int(action.shape[1]) - 1, fi - 1))
            raw_np = np.asarray(tok, dtype=np.float32)
            k = fi % RING
            pin_in[k].copy_(torch.from_numpy(raw_np))          # H2D from pinned memory, non-blocking (no stream sync)
            raw = pin_in[k].to(action.device, non_blocking=True)
            action[0, row, :] = net.cond_action(raw[None, None])[0, 0].to(action.dtype)
            frame_meta[fi] = (t_src, q14, raw_np)
            box["inflight"] += 1
            af0[0] = fi

        try:
            with torch.no_grad():
                net.rollout(kw["z_hist"], action, kw["state"], kw["frame_idx"],
                            seed_latents=kw["seed_latents"], seed_contact=kw["seed_contact"],
                            compiled=True, on_commit=on_commit, pending_fn=pending, on_admit=on_admit, wait_fn=wait)
            if sess["alive"]:
                print(f"[demo] generation exhausted TOTAL={TOTAL} committed frames -> reseeding", flush=True)
                box["reset"] = box["reset"] or "same"
            else:
                print(f"[demo] rollout unwound at frame {af0[0]} (session ended while idle)", flush=True)
        except _Abort:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"[demo] rollout FAILED: {e!r}", flush=True)
            loop.call_soon_threadsafe(push, {"type": "error", "error": repr(e)})
        finally:
            if dth is not None:
                dq.put(None)            # a leaked worker would block on get() for the process lifetime
                dth.join(timeout=10)

    async def stats_loop() -> None:
        """2 Hz side channel for everything that is NOT per-frame."""
        import statistics as _stx
        try:
            while True:
                await asyncio.sleep(0.5)
                push({"type": "stats", "mem": box["mem"], "bank": box["bank"], "committed": box["committed"],
                      "enq": box["enqueued"], "drop": box["dropped"], "n_memory": int(net.n_memory),
                      "mem_frames": box["mem_frames"][:16], "ctx_hist": box["ctx_hist"],
                      "dyn_ms": (_stx.median(box["dyn_ms"]) if box["dyn_ms"] else None),
                      "dec_ms": (_stx.median(box["dec_ms"]) if box["dec_ms"] else None),
                      "prologue_ms": box["prologue_ms"], "ws_drop": box["ws_drop"],
                      "compiled": compile_status(net, _dec_tok(M))})
        except asyncio.CancelledError:
            return

    recv_task = asyncio.create_task(recv_loop())
    tick_task = asyncio.create_task(ticker())
    stats_task = asyncio.create_task(stats_loop())
    torch.cuda.reset_peak_memory_stats()
    ep = EP if EP >= 0 else None
    th = None
    try:
        while sess["alive"]:
            box["reset"], box["state"] = None, "starting"
            box["enqueued"], box["dropped"], box["inflight"] = 0, 0, 0
            # Snapshot the mode for this generation and rebuild the key table from it (held keys survive a
            # same-mode reseed; mode_switch released them).
            box["mode"], box["ctrl"] = MODE["cur"], ctrl_of(MODE["cur"])
            held = {k: keys.get(k, 0) for k in key_names(box["ctrl"])}
            keys.clear()
            keys.update(held)
            while not sess["gate"].empty():           # leftover tokens would abort the new generation at once
                sess["gate"].get_nowait()
            spawn = (box["mouse"]["arm"] or "*") if box["mode"] == "mouse" else None
            hist = box.setdefault("scene_hist", [])   # tasks shown this session (New scene never repeats the current one)
            S = await loop.run_in_executor(None, lambda: seed_episode(ep, box["task"], box["seed_mode"], spawn=spawn,
                                                                     history=hist, first=box.get("n_new", 0) <= 1))
            box["S"] = S
            hist.append(S.get("task"))
            if getattr(ws, "client_state", None) is not None and ws.client_state.name != "CONNECTED":
                break                                 # viewer left during the (re)seed
            hello_msg = {
                "type": "hello", "ep": S["ep"], "asset_id": S["aid"], "task": S["task"],
                "seed_mode": S["seed_mode"], "pad": S["kw"].get("pad", 0), "seed_frames_const": SEED_FRAMES,
                "load_ms": S.get("load_ms"), "seed_cut": S.get("seed_cut"), "seed_cut_info": S.get("seed_cut_info"),
                "mouse": {"axes": [("+" if sg > 0 else "-") + "xy"[ax] for ax, sg in MOUSE_AXES],
                          "box": {a_: [b_[0].round(3).tolist(), b_[1].round(3).tolist()] for a_, b_ in MOUSE_BOX.items()},
                          "dz": MOUSE_DZ, "rot_step": MOUSE_ROT_STEP, "max_dp": MOUSE_MAX_DP, "smooth": MOUSE_SMOOTH,
                          "arm": box["mouse"]["arm"], "active": box["mouse"]["active"], "spawn": MOUSE_SPAWN,
                          "track": "grace", "grace_s": MOUSE_GRACE_S},
                "compiled": compile_status(net, _dec_tok(M)),
                "q0": [float(x) for x in S["kw"]["state"][0, S["kw"]["n_seed"]].float().cpu()],   # the seed joint state
                "model": M["model_id"], "tasks": M["tasks"][:TASK_LIST_MAX],
                "view_order": [v for v in VIEW_ORDER if v in net.obs_keys],
                "n_seed": S["kw"]["n_seed"], "total": TOTAL, "views": list(net.obs_keys),
                "grip_step": GRIP_STEP, "action_hz": ACTION_HZ, "queue_max": QUEUE_MAX,
                "ee_pos_step": EE_POS_STEP, "ee_rot_step": EE_ROT_STEP, "ik_damp": IK_DAMP,
                "pipelined": M["pipe"] is not None,
                "ctrl": [[c[0], c[1], str(c[2]), c[-1]] for c in box["ctrl"]],
                "ctrl_mode": box["mode"], "modes": list(MODES),
                "ee_box": {a: [list(map(float, b[0])), list(map(float, b[1]))] for a, b in EE_BOX.items()},
                "ik": {"iters": HAND_IK_ITERS, "max_dq": HAND_MAX_DQ, "w_pos": HAND_W_POS, "w_rot": HAND_W_ROT,
                       "w_post": HAND_W_POST, "w_smooth": HAND_W_SMOOTH, "damp": HAND_IK_DAMP},
                "tcp": HAND_TCP.tolist(),
                "dyn_flow_steps": int(net.dyn_flow_steps), "decode_steps": DECODE_STEPS,
                "grip_range": [[float(M["lo"][L_GRIP]), float(M["hi"][L_GRIP])], [float(M["lo"][R_GRIP]), float(M["hi"][R_GRIP])]]}
            _LIVE["hello"] = hello_msg
            await ws.send_text(json.dumps(hello_msg))
            # the gate starts EMPTY: the seed frame is shown and the rollout idles until the first input
            th = threading.Thread(target=run_generation, args=(S,), daemon=True)
            th.start()
            while th.is_alive() or outj or slot[0] is not None:
                try:
                    await asyncio.wait_for(wake.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                wake.clear()                              # clear BEFORE draining: no lost wake-ups
                try:
                    while outj:                           # control JSON first (small, never dropped)
                        await ws.send_text(outj.popleft())
                    b = slot[0]
                    if b is None:
                        continue
                    slot[0] = None
                    await ws.send_bytes(b)
                except (RuntimeError, ConnectionError, OSError, WebSocketDisconnect) as e:
                    print(f"[demo] viewer gone during send ({type(e).__name__}); ending session", flush=True)
                    sess["alive"] = False
                    _gate_put(sess, KILL)
                    break
            await loop.run_in_executor(None, th.join, 30)
            if not sess["alive"] or box["reset"] is None:
                break
            ep = None if box["reset"] == "new" else S["ep"]
            if box["reset"] == "new":
                box["n_new"] = box.get("n_new", 0) + 1   # the 1st New scene = the visitor's pick (honoured as is)
                if box["n_new"] == 1:
                    box["scene_hist"] = []                 # the connect-time pre-seed is not a scene the visitor chose
            if EP >= 0:
                ep = EP                                # DEMO_EP pins both Reset and New episode
            if box.get("ep_pin") is not None:
                ep = box.pop("ep_pin")
            print(f"[demo] RESET ({box['reset']}) -> reseeding", flush=True)
    except Exception:  # noqa: BLE001
        import traceback
        print("[demo] session FAILED:\n" + traceback.format_exc(), flush=True)
    finally:
        sess["alive"] = False
        _gate_put(sess, KILL)
        recv_task.cancel()
        tick_task.cancel()
        stats_task.cancel()
        if th is not None:
            await loop.run_in_executor(None, th.join, 60)
            if th.is_alive():
                print("[demo] WARNING: rollout thread did not exit; holding the GPU lock", flush=True)
                return                                # never hand the GPU to a new session while an old rollout owns it
        if _LIVE["sess"] is sess:
            _LIVE["sess"] = None
        print(f"[demo] session end: {box['committed']} committed frames, peak_mem "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        _GPU.release()
