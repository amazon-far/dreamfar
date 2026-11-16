"""Headless check of the demo server: connect, press keys for a few seconds, save the received frames.

    python -m dreamfar.demo.ws_client --url ws://127.0.0.1:8000/ws --seconds 8 --out outputs/demo_check
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import time

import websockets

HDR = struct.Struct(">diffi6fi")      # t_client, frame_idx, g_left, g_right, aq, L_pos(3), R_pos(3), in-flight depth


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--keys", default="w,r", help="keys held during the session (comma-separated)")
    p.add_argument("--out", default="outputs/demo_check")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_frames, hello = 0, None
    async with websockets.connect(a.url, max_size=32 * 1024 * 1024) as ws:
        t_end = time.time() + a.seconds
        held = {k: 1 for k in a.keys.split(",") if k}
        last_send = 0.0
        while time.time() < t_end:
            if time.time() - last_send > 0.1:
                await ws.send(json.dumps({"type": "keys", "keys": held, "t_client": time.time() * 1000}))
                last_send = time.time()
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(msg, bytes):
                t_client, fi, gl, gr, aq, *rest = HDR.unpack(msg[:HDR.size])
                if n_frames % 10 == 0:
                    open(os.path.join(a.out, f"frame_{fi:05d}.jpg"), "wb").write(msg[HDR.size:])
                n_frames += 1
            else:
                m = json.loads(msg)
                if m.get("type") == "hello":
                    hello = m
                    print(f"[hello] task={m.get('task')!r} asset={m.get('asset_id')} mode={m.get('ctrl_mode')} "
                          f"n_seed={m.get('n_seed')}", flush=True)
        await ws.send(json.dumps({"type": "keys", "keys": {}, "t_client": time.time() * 1000}))
    print(f"[done] hello={'yes' if hello else 'no'} frames={n_frames} in {a.seconds:.0f}s -> {a.out}", flush=True)
    assert hello is not None and n_frames > 0, "no hello or no frames received"


if __name__ == "__main__":
    asyncio.run(main())
