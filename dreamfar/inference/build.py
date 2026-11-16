"""Build a DynamicsModule for inference from a checkpoint + its config, and decode latents."""
from __future__ import annotations

import os

import torch
from omegaconf import OmegaConf

from dreamfar.data.normalizer import LinearNormalizer
from dreamfar.dynamics.module import DynamicsModule

DECODE_NOISE_SEED = 1234   # one fixed decoder noise per process -> no per-frame flicker


def load_algorithm_cfg(config_path: str, overrides: dict | None = None):
    """A run's ``.hydra/config.yaml`` (full tree) or a bare ``algorithm`` node -> the algorithm cfg."""
    cfg = OmegaConf.load(config_path)
    algo = cfg.algorithm if "algorithm" in cfg else cfg
    OmegaConf.resolve(cfg)
    algo = OmegaConf.create(OmegaConf.to_container(algo, resolve=True))
    algo.grad_checkpoint = False                      # training-only activation checkpointing
    for k, v in (overrides or {}).items():
        OmegaConf.update(algo, k, v, merge=False)
    return algo


def build_model(ckpt_path: str, config_path: str, device: str = "cuda:0", ema: bool = True,
                overrides: dict | None = None) -> DynamicsModule:
    """Instantiate from `config_path`, load `dynamics.*` (EMA weights when present) and the
    normalizer baked into the checkpoint. The checkpoint's tensor names/shapes must match exactly."""
    torch.set_float32_matmul_precision("high")        # TF32 matmuls, as in training
    cfg = load_algorithm_cfg(config_path, overrides)
    model = DynamicsModule(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k.replace("._orig_mod.", "."): v for k, v in ck["state_dict"].items()}
    dyn = {k[len("dynamics."):]: v for k, v in sd.items() if k.startswith("dynamics.")}
    live = model.dynamics.state_dict()
    only_model = sorted(set(live) - set(dyn))
    only_ckpt = sorted(set(dyn) - set(live))
    bad = sorted(k for k in set(live) & set(dyn) if tuple(live[k].shape) != tuple(dyn[k].shape))
    if only_model or only_ckpt or bad:
        raise SystemExit(f"architecture mismatch vs {os.path.basename(ckpt_path)}: "
                         f"only in model {only_model[:5]}, only in ckpt {only_ckpt[:5]}, shapes {bad[:5]}")
    model.dynamics.load_state_dict(dyn, strict=True)
    src = "live"
    if ema and ck.get("ema") is not None:
        ema_sd = ck["ema"]
        model.dynamics.load_state_dict({n: ema_sd.get(n, live[n]) for n in live}, strict=True)
        src = f"EMA({sum(1 for n in live if n in ema_sd)}/{len(live)})"
    nz = LinearNormalizer()
    nz.load_state_dict({k[len("normalizer."):]: v for k, v in sd.items() if k.startswith("normalizer.")})
    model.set_normalizer(nz)
    model = model.to(device).eval()
    dtype = model.apply_infer_weights_dtype()
    print(f"[build] {os.path.basename(ckpt_path)} step={ck.get('global_step')} weights={src} "
          f"{len(live)} tensors, dynamics dtype={str(dtype).replace('torch.', '')}", flush=True)
    return model


def decode_noise(model, device: str) -> torch.Tensor:
    """The fixed decoder noise (1, C_out, H, W) shared by every decoded frame."""
    g = torch.Generator(device=device).manual_seed(DECODE_NOISE_SEED)
    res = int(model.image_size)
    return torch.randn((1, model.tokenizer.decoder.out_dim, res, res), generator=g, device=device)


@torch.no_grad()
def decode_frame(tokenizer, z_frame: torch.Tensor, n_views: int, base_noise: torch.Tensor,
                 steps: int) -> torch.Tensor:
    """One view-stacked latent (1,C,V*h,w) -> (V,3,H,W) float in [0,1] (one batched decoder call)."""
    C, Vh, w = z_frame.shape[-3:]
    h = Vh // n_views
    zv = z_frame.reshape(1, C, n_views, h, w)[0].permute(1, 0, 2, 3).contiguous()   # (V,C,h,w)
    noise = base_noise.expand(n_views, -1, -1, -1).contiguous()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rgb = tokenizer.decode(zv, steps=steps, noise=noise).float()
    return ((rgb + 1) / 2).clamp(0, 1)


@torch.no_grad()
def decode_sequence(model, z_seq: torch.Tensor, base_noise: torch.Tensor, steps: int,
                    chunk: int = 16) -> dict:
    """(1,T,C,V*h,w) latents -> {view: (T,3,H,W) float [0,1] on CPU}."""
    z_flat = z_seq[0]                                                 # (T,C,V*h,w)
    out = {}
    for k, zv in zip(model.obs_keys, model.split_views(z_flat)):
        rc = []
        for s in range(0, zv.shape[0], chunk):
            zc = zv[s:s + chunk]
            noise = base_noise.expand(zc.shape[0], -1, -1, -1).contiguous()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rc.append(model.tokenizer.decode(zc, steps=steps, noise=noise).float())
        out[k] = ((torch.cat(rc, 0) + 1) / 2).clamp(0, 1).cpu()
    return out
