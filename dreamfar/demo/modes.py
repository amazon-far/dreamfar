"""Input-mode selection for the demo server: the pure, torch-free part.

One model instance serves two switchable input modes per session: `keyboard` (the eefik key table:
EE translation / rotation / gripper keys -> differential IK) and `mouse` (pointer over the TOP view ->
the selected arm's table-plane x,y, wheel -> z, MCTRL_SPEC keys -> orientation, Space -> gripper).
Protocol on /ws (text JSON):
    client -> server   {"type":"mode","mode":"keyboard"|"mouse","task":"<name>"|"","new_episode":true|false}
    server -> client   {"type":"mode","mode":<canonical>,"ok":true,"reset":"ack"|"reset_same"|"reset_new"}
                       then (unless "ack") a fresh `hello` whose ctrl_mode / ctrl describe the new mode.
Canonical names inside the server: `eefik` (keyboard) and `mouse`. A switch always goes through the
server's reset path, so the generation thread only ever reads the mode at (re)start.
"""
from __future__ import annotations

MODES = ("keyboard", "mouse")                 # the public menu (hello "modes"), in display order
_ALIAS = {"keyboard": "eefik", "eefik": "eefik", "mouse": "mouse"}


def normalize_mode(name: object) -> str | None:
    """'keyboard' / 'eefik' -> 'eefik'; 'mouse' -> 'mouse'; anything else -> None."""
    if not isinstance(name, str):
        return None
    return _ALIAS.get(name.strip().lower())


# Raw key tables: (up, dn, arm, kind, axis, step TAG, label); kind in {t=translate, r=rotate, g=grip}.
# server.py attaches the env-dependent step sizes. The FIRST key of each pair always INCREASES that
# quantity, and the pairing is mirrored between arms so a control means the same thing on both.
#   LEFT   translate  W/S fwd-back   A/D left-right   Q/E up-down
#          rotate     1/Z roll       2/X pitch        3/C yaw
#          gripper    R/F open-close
#   RIGHT  translate  I/K fwd-back   J/L left-right   U/O up-down
#          rotate     7/M roll       8/, pitch        9/. yaw
#          gripper    P/; open-close
# tag p = EE position step, o = EE rotation step, c = grip step.
ECTRL_SPEC = [
    ("w", "s", "L", "t", 0, "p", "L_fwd"),
    ("a", "d", "L", "t", 1, "p", "L_left"),
    ("q", "e", "L", "t", 2, "p", "L_up"),
    ("1", "z", "L", "r", 0, "o", "L_roll"),
    ("2", "x", "L", "r", 1, "o", "L_pitch"),
    ("3", "c", "L", "r", 2, "o", "L_yaw"),
    ("r", "f", "L", "g", 0, "c", "L_grip_open"),
    ("i", "k", "R", "t", 0, "p", "R_fwd"),
    ("j", "l", "R", "t", 1, "p", "R_left"),
    ("u", "o", "R", "t", 2, "p", "R_up"),
    ("7", "m", "R", "r", 0, "o", "R_roll"),
    ("8", ",", "R", "r", 1, "o", "R_pitch"),
    ("9", ".", "R", "r", 2, "o", "R_yaw"),
    ("p", ";", "R", "g", 0, "c", "R_grip_open"),
]

# Mouse-mode keys: ORIENTATION of the SELECTED arm (arm "*"; the pointer picks it); tag m = the mouse
# rotation step per 10 Hz tick while held (at least one step per tap). W/Q roll +/-, S/A pitch +/-,
# X/Z yaw +/- -- world-frame (arm base) rotations about the TCP. Translation is the pointer (x,y) +
# wheel (z); the gripper is Space / middle click (a toggle, not a key pair).
MCTRL_SPEC = [
    ("w", "q", "*", "r", 0, "m", "roll"),
    ("s", "a", "*", "r", 1, "m", "pitch"),
    ("x", "z", "*", "r", 2, "m", "yaw"),
]

_TABLES: dict[str, list] = {"eefik": ECTRL_SPEC, "mouse": MCTRL_SPEC}


def ctrl_table(mode: str) -> list:
    """The raw key table a (canonical) mode exposes to the page."""
    return _TABLES.get(mode, [])


def key_names(table: list) -> list:
    """Every key the page may report held for this table (the session's `keys` dict)."""
    return [k for c in table for k in c[:2]]


def mode_transition(cur: str, new: str, new_episode: bool) -> str:
    """'ack' (same mode, same episode), 'reset_same' (switch modes, keep the seed episode) or
    'reset_new' (a new episode, whether or not the mode changes)."""
    if new_episode:
        return "reset_new"
    return "ack" if new == cur else "reset_same"
