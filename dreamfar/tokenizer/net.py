"""SsddTokenizer: SSDD f16 encoder (unit-sphere latent) + U-ViT flow-matching decoder.

``encode(x) -> z`` (B,16,H/16,W/16) unit-norm; ``decode(z, steps) -> x`` in [-1,1];
``forward(x)`` in train mode returns a :class:`TrainOut` with the flow loss and ``x0_pred``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from ssdd.flow import FlowMatchingTrainer, FMEulerSampler

from .decoder import UViTDecoder
from .encoder import SsddEncoder


@dataclass
class TrainOut:
    x0_gt: torch.Tensor
    x0_pred: torch.Tensor
    xt: torch.Tensor
    t: torch.Tensor
    z: torch.Tensor
    noise: torch.Tensor
    losses: dict = field(default_factory=dict)


class SsddTokenizer(nn.Module):
    def __init__(
        self,
        encoder_kwargs: Optional[dict] = None,
        decoder_size: str = "M",
        z_dim: int = 16,
        fm_trainer: Optional[dict] = None,
        fm_sampler: Optional[dict] = None,
        encoder_checkpoint: Optional[str] = None,
        decoder_checkpoint: Optional[str] = None,
        decoder_z_scale: Optional[float] = None,
        decoder_attn_norm: Optional[str] = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = SsddEncoder(z_dim=z_dim, **dict(encoder_kwargs or {}))
        self.decoder = UViTDecoder.make(decoder_size, z_dim=z_dim, attn_norm=decoder_attn_norm,
                                        z_scale=decoder_z_scale)
        # The latent grid must DIVIDE the decoder mid-block grid (nearest upsample = exact repeat).
        down_scale = 2 ** (len(self.decoder.ch_level) - 1)
        assert self.encoder.patch_size % down_scale == 0, (
            f"encoder patch_size {self.encoder.patch_size} is not a multiple of the decoder "
            f"down_scale {down_scale}")
        self.fm_trainer = FlowMatchingTrainer(**(fm_trainer or {}))
        self.fm_sampler = FMEulerSampler(**(fm_sampler or {}))
        if encoder_checkpoint:
            self.encoder.load_pretrained(encoder_checkpoint)
        if decoder_checkpoint:
            r = self.decoder.load_pretrained(decoder_checkpoint)
            print(f"[ssdd-dual-init] decoder <- {decoder_checkpoint}: loaded {r['loaded']}/"
                  f"{r['ckpt_tensors']} ckpt tensors into {r['own_keys']} keys; left at init: "
                  f"{len(r['missing_qk_norm'])} qk-norm + {r['missing_own_only']}; "
                  f"z_scale init {r['z_scale']}", flush=True)

    @property
    def latent_dim(self) -> int:
        return int(self.z_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, steps: Optional[int] = None, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, zH, zW = z.shape
        H, W = zH * self.encoder.patch_size, zW * self.encoder.patch_size
        return self.fm_sampler.sample(
            self.decoder, self.fm_trainer,
            shape=(B, self.decoder.out_dim, H, W),
            steps=steps, fn_kwargs={"z": z}, noise=noise,
        )

    def forward(self, gt_x: torch.Tensor, steps: Optional[int] = None):
        z = self.encode(gt_x)
        if not self.training:
            return self.decode(z, steps=steps)
        # training: flow-matching loss + predicted clean image (for the LPIPS/REPA aux losses)
        t = self.fm_trainer.sample_t(gt_x.shape[0], device=gt_x.device)
        diff_loss, (x_t, noise, noise_t, v_pred) = self.fm_trainer.loss(
            self.decoder, x=gt_x, t=t, fn_kwargs={"z": z}
        )
        x0_pred = self.fm_trainer.step(x_t, v_pred, noise_t)  # -> t=0 (clean) estimate
        return TrainOut(x0_gt=gt_x, x0_pred=x0_pred, xt=x_t, t=t, z=z, noise=noise,
                        losses={"diffusion": diff_loss})


# --------------------------------------------------------------------------- #
# Construction from a tokenizer `algorithm` config + loading a Lightning checkpoint
# --------------------------------------------------------------------------- #
def build_tokenizer(tok_cfg) -> SsddTokenizer:
    """Build the net from a tokenizer ``algorithm`` DictConfig; no SSDD safetensors are loaded
    (the trained weights come from the Lightning ckpt, see :func:`load_tokenizer`)."""
    assert str(tok_cfg.get("encoder_type", "ssdd_native")) == "ssdd_native", tok_cfg.get("encoder_type")
    assert str(tok_cfg.get("decoder_levels", "native")).lower() == "native", tok_cfg.get("decoder_levels")
    return SsddTokenizer(
        encoder_kwargs=dict(tok_cfg.get("encoder", {})),
        decoder_size=tok_cfg.decoder_size,
        z_dim=tok_cfg.z_dim,
        fm_trainer=dict(tok_cfg.get("fm_trainer", {})),
        fm_sampler=dict(tok_cfg.get("fm_sampler", {})),
        encoder_checkpoint=None,
        decoder_z_scale=tok_cfg.get("decoder_z_scale", None),
        decoder_attn_norm=tok_cfg.get("decoder_attn_norm", None),
    )


class _EncStub(nn.Module):
    """Placeholder encoder carrying only ``patch_size`` (decode needs it for the output H,W)."""


def load_tokenizer(ckpt_path: str, cfg_path: str, drop_encoder: bool = False) -> SsddTokenizer:
    """Rebuild the frozen tokenizer from a Lightning ckpt (``net.*`` weights) + its algorithm yaml.

    Load is ``strict=False`` with the original accounting: every ``encoder.*`` key must be present
    (asserted); decoder mismatches only warn. ``drop_encoder`` swaps the encoder for a patch_size stub."""
    from omegaconf import OmegaConf

    tok = build_tokenizer(OmegaConf.load(cfg_path).algorithm)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k.replace("._orig_mod.", "."): v for k, v in ckpt["state_dict"].items()}
    net_sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
    missing, unexpected = tok.load_state_dict(net_sd, strict=False)
    enc_missing = [k for k in missing if k.startswith("encoder.")]
    assert not enc_missing, f"tokenizer encoder weights missing from ckpt: {enc_missing[:5]}"
    print(f"[tokenizer] loaded {ckpt_path} (step {ckpt.get('global_step', '?')}); "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    dec_bad = [k for k in list(missing) + list(unexpected) if k.startswith("decoder.")]
    if dec_bad:
        print(f"[tokenizer] WARNING: {len(dec_bad)} decoder tensors did not match the ckpt "
              f"(e.g. {dec_bad[:3]}); check decoder_z_scale / decoder_attn_norm / decoder_size.", flush=True)
    for p in tok.parameters():
        p.requires_grad = False
    tok.eval()
    if drop_encoder:
        stub = _EncStub()
        stub.patch_size = tok.encoder.patch_size
        tok.encoder = stub
    return tok
