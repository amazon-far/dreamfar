"""Encode raw episodes with the frozen tokenizer into latent shards (the dynamics training input).

Raw layout: ``<raw_root>/index.json`` {"fps", "episodes": [{"id", "task_name", "size"}]} and per episode
``<raw_root>/<id>/{top,left,right}.mp4`` + ``states_actions.bin`` (float64 (size, 28)).
Output per shard ``<out>/shard_XX/``: latents.bin (N,V,C,h,w) float16, states_actions.bin (N,28) float32,
states_actions_ee.bin (N,40) float32 (FK pass), index.json.

    python -m dreamfar.data.precompute_latents --raw-root <raw> --out <latent_root> \
        --tokenizer-ckpt <ckpt> --tokenizer-config <cfg> --urdf yam.urdf [--shard-episodes N] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dreamfar.data.ee_fk import load_urdf, write_ee_bin
from dreamfar.data.video import decode_frames, letterbox_resize
from dreamfar.tokenizer.net import load_tokenizer

VIEWS = ("top", "left", "right")  # fixed stored order -> latent view axis
SA_COLS = 28


def build_tokenizer(ckpt_path: str, config_path: str, device: torch.device,
                    compile_encoder: bool = True, channels_last: bool = True):
    """Frozen tokenizer in eval mode; channels_last + torch.compile on the encoder (encode is the whole workload)."""
    tok = load_tokenizer(ckpt_path, config_path, drop_encoder=False)
    for p in tok.parameters():
        p.requires_grad = False
    tok.eval().to(device)
    if channels_last and device.type == "cuda":
        tok.encoder = tok.encoder.to(memory_format=torch.channels_last)
    if compile_encoder:
        tok.encoder = torch.compile(tok.encoder, mode="max-autotune")
    assert not tok.training, "tokenizer is not in eval mode"
    return tok


class _EpisodeFrames(Dataset):
    """One episode's FULL decoded 3-view frames (letterboxed) + states_actions, per index (DataLoader
    workers decode ahead of the GPU)."""

    def __init__(self, raw_root: str, episodes, resolution: int):
        self.raw_root = raw_root
        self.episodes = list(episodes)
        self.hw = (resolution, resolution)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i: int) -> dict:
        ep = self.episodes[i]
        d = os.path.join(self.raw_root, ep["id"])
        rec = {"asset_id": ep["id"], "size": int(ep["size"]), "task_name": ep.get("task_name", "")}
        try:
            idx = list(range(int(ep["size"])))
            arr = np.fromfile(os.path.join(d, "states_actions.bin"), dtype="<f8").reshape(-1, SA_COLS)
            sa = torch.from_numpy(arr.astype(np.float32, copy=True))                    # (N,28) fp32
            imgs = {c: letterbox_resize(decode_frames(os.path.join(d, f"{c}.mp4"), idx), self.hw)
                    for c in VIEWS}                                                      # (N,H,W,C) u8
            n = sa.shape[0]
            for c in VIEWS:
                if imgs[c].shape[0] != n:
                    rec["error"] = f"len mismatch {c}={imgs[c].shape[0]} vs sa={n}"
                    return rec
            rec["images"] = imgs
            rec["sa"] = sa
            rec["size"] = int(n)
        except Exception as e:  # noqa: BLE001 -- one bad episode must not kill the shard
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec


def _identity_collate(batch):
    return batch[0]  # batch_size=1


@torch.no_grad()
def encode_episode(tok, imgs: dict, device, chunk: int, channels_last: bool = True) -> torch.Tensor:
    """imgs {cam:(N,H,W,C) uint8} -> (N, V, C, h, w) fp16 unit-sphere latents. Per view: uint8 -> float/255
    -> [-1,1] -> encode, all views folded into the batch; every forward is padded to a fixed `chunk` so
    torch.compile sees one static shape."""
    assert not torch.is_grad_enabled(), "@torch.no_grad() did not take effect"
    x_u8 = torch.cat([imgs[cam] for cam in VIEWS], dim=0)  # (V*N,H,W,C) uint8
    VN = x_u8.shape[0]
    zs = []
    for c0 in range(0, VN, chunk):
        xb = x_u8[c0:c0 + chunk].to(device, non_blocking=True)
        nb = xb.shape[0]
        if nb < chunk:                                    # pad the ragged tail to a static shape
            xb = torch.cat([xb, xb.new_zeros((chunk - nb, *xb.shape[1:]))], dim=0)
        xb = xb.permute(0, 3, 1, 2).float().div_(255.0).mul_(2.0).sub_(1.0)  # (chunk,3,H,W) [-1,1]
        if channels_last and device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            z = tok.encode(xb)                            # (chunk,C,h,w) unit-sphere
        zs.append(z[:nb].float().cpu())                   # drop pad rows
    z = torch.cat(zs, dim=0)                              # (V*N,C,h,w)
    N = z.shape[0] // len(VIEWS)
    z = z.view(len(VIEWS), N, *z.shape[1:])               # (V,N,C,h,w)
    return z.permute(1, 0, 2, 3, 4).contiguous().half()   # (N,V,C,h,w) fp16


def encode_shard(shard_dir: str, raw_root: str, episodes, tok, device, args, latent_shape, fps: int) -> int:
    """Encode ``episodes`` into ``shard_dir`` (latents + joint bin + index.json), return the episode count."""
    os.makedirs(shard_dir, exist_ok=True)
    loader = DataLoader(
        _EpisodeFrames(raw_root, episodes, args.resolution),
        batch_size=1, num_workers=args.decode_workers, shuffle=False,
        collate_fn=_identity_collate, prefetch_factor=(2 if args.decode_workers else None),
        multiprocessing_context=("fork" if args.decode_workers else None),
        persistent_workers=False,
    )
    done, rows = [], 0
    t0 = time.perf_counter()
    with open(os.path.join(shard_dir, "latents.bin"), "wb") as lat_f, \
            open(os.path.join(shard_dir, "states_actions.bin"), "wb") as sa_f:
        for rec in loader:
            if "error" in rec:
                print(f"[{os.path.basename(shard_dir)}] SKIP {rec['asset_id']}: {rec['error']}", flush=True)
                continue
            lat = encode_episode(tok, rec["images"], device, args.encode_chunk,
                                 channels_last=not args.no_channels_last)                # (N,V,C,h,w) fp16
            sa = rec["sa"].to(torch.float32).contiguous()                               # (N,28)
            assert lat.shape[0] == sa.shape[0] == rec["size"]
            lat.numpy().tofile(lat_f)
            sa.numpy().tofile(sa_f)
            done.append({"asset_id": rec["asset_id"], "start": rows, "size": int(rec["size"]),
                         "task_name": rec["task_name"]})
            rows += int(rec["size"])
            if len(done) % args.log_every == 0:
                dt = time.perf_counter() - t0
                print(f"[{os.path.basename(shard_dir)}] {len(done)}/{len(episodes)} eps | {rows} rows | "
                      f"{rows / max(dt, 1e-6):.0f} frame/s", flush=True)
    if not done:
        print(f"[{os.path.basename(shard_dir)}] no episode encoded; index.json not written", flush=True)
        return 0
    idx = {
        "views": list(VIEWS), "latent_shape": list(latent_shape), "latent_dtype": "float16",
        "sa_cols": SA_COLS, "sa_dtype": "float32", "state_dim": SA_COLS // 2,
        "fps": fps, "frame_skip": 1, "resolution": args.resolution,
        "tokenizer_ckpt": args.tokenizer_ckpt, "tokenizer_config": args.tokenizer_config,
        "n_episodes": len(done), "n_frames": rows, "episodes": done,
    }
    with open(os.path.join(shard_dir, "index.json"), "w") as f:
        json.dump(idx, f)
    return len(done)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True)
    p.add_argument("--out", required=True, help="latent root; shard_XX dirs are created under it")
    p.add_argument("--tokenizer-ckpt", required=True)
    p.add_argument("--tokenizer-config", required=True)
    p.add_argument("--urdf", required=True, help="YAM URDF for the EE-pose FK pass")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--shard-episodes", type=int, default=512, help="episodes per shard dir")
    p.add_argument("--shard-index", type=int, default=0, help="this process handles shards [shard_index::num_shards]")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--decode-workers", type=int, default=4, help="DataLoader workers decoding video")
    p.add_argument("--encode-chunk", type=int, default=256, help="frames per encoder forward (fixed; padded)")
    p.add_argument("--log-every", type=int, default=50, help="episodes between progress logs")
    p.add_argument("--no-compile", action="store_true", help="disable torch.compile of the encoder")
    p.add_argument("--no-channels-last", action="store_true", help="disable channels_last conv layout")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True          # static shapes -> autotune conv algos
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)

    index = json.load(open(os.path.join(args.raw_root, "index.json")))
    episodes = index["episodes"]
    fps = int(index.get("fps", 30))
    shards = [episodes[i:i + args.shard_episodes] for i in range(0, len(episodes), args.shard_episodes)]
    mine = list(range(args.shard_index, len(shards), args.num_shards))
    print(f"[precompute] {len(episodes)} episodes -> {len(shards)} shards; this process: {len(mine)}", flush=True)

    tok = build_tokenizer(args.tokenizer_ckpt, args.tokenizer_config, device,
                          compile_encoder=not args.no_compile, channels_last=not args.no_channels_last)
    # probe the latent geometry (also warms the compiled static-shape encoder)
    probe = {c: torch.zeros((1, args.resolution, args.resolution, 3), dtype=torch.uint8) for c in VIEWS}
    latent_shape = list(encode_episode(tok, probe, device, args.encode_chunk,
                                       channels_last=not args.no_channels_last).shape[1:])   # [V,C,h,w]
    print(f"[precompute] latent_shape={latent_shape}", flush=True)
    urdf = load_urdf(args.urdf)

    t0 = time.perf_counter()
    for si in mine:
        shard_dir = os.path.join(args.out, f"shard_{si:02d}")
        if os.path.exists(os.path.join(shard_dir, "index.json")):
            print(f"[precompute] {shard_dir} already complete; skipping", flush=True)
            continue
        n = encode_shard(shard_dir, args.raw_root, shards[si], tok, device, args, latent_shape, fps)
        if n:
            frames = write_ee_bin(shard_dir, urdf)
            print(f"[precompute] {shard_dir}: {n} eps / {frames} frames (+ states_actions_ee.bin)", flush=True)
    print(f"[precompute] DONE in {(time.perf_counter() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
