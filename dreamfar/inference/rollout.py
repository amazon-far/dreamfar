"""Roll the world model out on a held-out episode, decode, score and write a [pred | GT] video.

    python -m dreamfar.inference.rollout --ckpt <dynamics.ckpt> --config <run config.yaml> \
        --latent-root <latent_root> --episode 0 --regime mid --frames 300 --out outputs/rollouts

Metrics (PSNR / SSIM / LPIPS per view, over the predicted frames only) go to <out>/<asset>_<regime>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import imageio.v2 as imageio
import numpy as np
import torch
from einops import rearrange

from dreamfar.inference.build import build_model, decode_noise, decode_sequence
from dreamfar.inference.episodes import list_episodes, load_episode
from dreamfar.inference.seeding import build_regime

# Canonical deployed inference settings (override the training-time validation values).
CANONICAL = {"dyn_flow_steps": 16, "infer_t_schedule": "logit_normal",
             "stream_ctx_noise": 0.2404252730845196}


def _metrics(pred: dict, gt: dict, lo: int, device: str) -> dict:
    from torchmetrics.functional import peak_signal_noise_ratio as psnr
    from torchmetrics.functional import structural_similarity_index_measure as ssim
    import lpips
    net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    out = {}
    for k in pred:
        p, g = pred[k][lo:].to(device), gt[k][lo:].to(device)
        with torch.no_grad():
            lp = net(p * 2 - 1, g * 2 - 1).flatten().mean()
        out[k] = {"psnr": float(psnr(p, g, data_range=1.0)), "ssim": float(ssim(p, g, data_range=1.0)),
                  "lpips": float(lp)}
    out["mean"] = {m: float(np.mean([out[k][m] for k in pred])) for m in ("psnr", "ssim", "lpips")}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True, help="the run's .hydra/config.yaml (or a bare algorithm yaml)")
    p.add_argument("--latent-root", required=True)
    p.add_argument("--heldout", default=None, help="held-out split json; default = every episode")
    p.add_argument("--contact-dir", default=None, help="contact sidecars (seeds the memory bank for mid-start)")
    p.add_argument("--episode", default="0", help="index into the sorted episode list, or an asset_id")
    p.add_argument("--regime", default="mid", choices=["cold", "dup", "mid"])
    p.add_argument("--seed-sec", type=float, default=10.0, help="mid-start seed length (s)")
    p.add_argument("--frames", type=int, default=300, help="episode cap in 10 Hz frames")
    p.add_argument("--decode-steps", type=int, default=2)
    p.add_argument("--weights-dtype", default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--no-ema", dest="ema", action="store_false")
    p.add_argument("--no-compile", dest="compiled", action="store_false")
    p.add_argument("--tokenizer-ckpt", default=None, help="override algorithm.tokenizer_ckpt of --config")
    p.add_argument("--tokenizer-config", default=None, help="override algorithm.tokenizer_config of --config")
    p.add_argument("--urdf", default=None, help="override algorithm.ee_urdf_path of --config")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    ov = {**CANONICAL, "infer_weights_dtype": a.weights_dtype}
    for k, v in (("tokenizer_ckpt", a.tokenizer_ckpt), ("tokenizer_config", a.tokenizer_config), ("ee_urdf_path", a.urdf)):
        if v:
            ov[k] = v
    model = build_model(a.ckpt, a.config, dev, ema=a.ema, overrides=ov)
    eps, meta = list_episodes(a.latent_root, a.heldout)
    ep = eps[int(a.episode)] if a.episode.isdigit() else next(e for e in eps if e.asset_id == a.episode)
    batch = load_episode(model, ep, meta, dev, cap=a.frames, contact_dir=a.contact_dir)
    z_gt = rearrange(model.stack_views(batch["latent"]), "t b c h w -> b t c h w")
    T = z_gt.shape[1]
    kw = build_regime(a.regime, model, z_gt, model.cond_action(batch["action"]), batch["state"],
                      batch["frame_idx"], contact=batch.get("contact"), seed_sec=a.seed_sec, fps=10)
    t0 = time.perf_counter()
    z_pred, _ = model.rollout(kw["z_hist"], kw["action"], kw["state"], kw["frame_idx"],
                              seed_latents=kw["seed_latents"], seed_contact=kw["seed_contact"],
                              compiled=a.compiled)
    torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    z_full = torch.cat([z_gt[:, :1], z_pred[:, kw["pad"]:kw["pad"] + T - 1]], dim=1)   # (1,T,...) incl. seed frame 0
    noise = decode_noise(model, dev)
    pred = decode_sequence(model, z_full.float(), noise, a.decode_steps)
    gt = decode_sequence(model, z_gt, noise, a.decode_steps)
    n_seed = kw["n_seed"] if a.regime == "mid" else 1
    m = _metrics(pred, gt, n_seed, dev)
    m.update(asset_id=ep.asset_id, task_name=ep.task_name, regime=a.regime, frames=T, n_seed=n_seed,
             rollout_s=round(dt, 2), fps=round((T - n_seed) / dt, 2))
    tag = f"{ep.asset_id[:8]}_{a.regime}"
    json.dump(m, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    rows = [torch.cat([pred[k], gt[k]], dim=-1) for k in model.obs_keys]            # (T,3,H,2W) per view
    vid = (torch.cat(rows, dim=-2).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
    imageio.mimwrite(os.path.join(a.out, tag + ".mp4"), list(vid), fps=10, macro_block_size=1)
    print(f"[rollout] {tag}: {T - n_seed} predicted frames in {dt:.1f}s ({m['fps']} fps); "
          f"PSNR {m['mean']['psnr']:.2f} SSIM {m['mean']['ssim']:.3f} LPIPS {m['mean']['lpips']:.3f} "
          f"-> {a.out}/{tag}.mp4", flush=True)


if __name__ == "__main__":
    main()
