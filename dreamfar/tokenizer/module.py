"""TokenizerModule: Lightning training wrapper for :class:`~dreamfar.tokenizer.net.SsddTokenizer`.

Loss = flow-matching + LPIPS + REPA (LingBot teacher, linear warm-up) + latent-uniformity hinge.
Optimizer: RAdamScheduleFree, two LR groups (pretrained encoder+decoder at ``backbone_lr``, the
REPA projector at ``base_lr``). Checkpoints hold the ScheduleFree averaged iterate (see ``state_dict``)."""
from __future__ import annotations

import math
import time
from typing import Optional

import lightning.pytorch as pl
import torch
from einops import rearrange
from omegaconf import DictConfig

from dreamfar.data.normalizer import LinearNormalizer, get_image_range_normalizer
from dreamfar.tokenizer.losses import AuxLosses
from dreamfar.tokenizer.net import SsddTokenizer


def sphere_uniform_const(z_dim: int, t: float) -> float:
    """``E[exp(-t*||z_i - z_j||^2)]`` for two independent uniform points on ``S^{z_dim-1}``
    (closed form via the modified Bessel function; c(8,2)=0.045843, c(16,2)=0.029919)."""
    s = 2.0 * t
    nu = z_dim / 2.0 - 1.0
    half = s / 2.0
    i_nu = 0.0
    for k in range(64):
        term = half ** (2 * k + nu) / (math.factorial(k) * math.gamma(k + nu + 1.0))
        i_nu += term
        if term < 1e-18 * max(i_nu, 1e-300):
            break
    return math.exp(-s) * math.gamma(z_dim / 2.0) * (2.0 / s) ** nu * i_nu


class TokenizerModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_keys = list(cfg.obs_keys)
        assert len(self.obs_keys) == 1, f"single-view image tokenizer; use one obs_key (got {self.obs_keys})"
        self.image_size = int(cfg.image_size)
        self.z_dim = int(cfg.z_dim)
        self.loss_coef = dict(cfg.loss_coef)
        self.unif_cfg = dict(cfg.get("latent_uniformity", {}) or {})
        self._resolve_uniform_const()
        self._build_model()

        # Fixed [0,1] -> [-1,1] image normalizer (same for every view); set_normalizer may replace it.
        self.normalizer = LinearNormalizer()
        for k in self.obs_keys:
            self.normalizer[k] = get_image_range_normalizer()
        # Recon-panel reservoirs: view -> [(recon, gt), ...] + frames-seen counters.
        self.validation_step_outputs: dict = {}
        self._val_seen: dict = {}
        self._panel_pool = int(cfg.get("val_panel_pool", 256))
        self._t_start: Optional[float] = None
        self._samples_seen = 0
        self.steps_per_epoch_estimate: Optional[int] = None

    def _resolve_uniform_const(self) -> None:
        """``uniform_const`` null -> derive c(z_dim, t); a number -> keep it but raise if it is >2% off."""
        if not bool(self.unif_cfg.get("enabled", False)):
            return
        t = float(self.unif_cfg.get("t", 2.0))
        derived = sphere_uniform_const(self.z_dim, t)
        given = self.unif_cfg.get("uniform_const", None)
        if given is None:
            self.unif_cfg["uniform_const"] = derived
            print(f"[uniformity] uniform_const derived for z_dim={self.z_dim} t={t}: {derived:.6f}", flush=True)
            return
        if abs(float(given) - derived) > 0.02 * derived:
            raise ValueError(
                f"latent_uniformity.uniform_const={given} is for a different sphere: on S^{self.z_dim - 1} "
                f"at t={t} it is {derived:.6f}. Set it to that value (or null to derive it).")

    # ===================== model construction ============================= #
    def _build_model(self) -> None:
        cfg = self.cfg
        self.net = SsddTokenizer(
            encoder_kwargs=dict(cfg.get("encoder", {})),
            decoder_size=cfg.decoder_size,
            z_dim=cfg.z_dim,
            fm_trainer=dict(cfg.get("fm_trainer", {})),
            fm_sampler=dict(cfg.get("fm_sampler", {})),
            encoder_checkpoint=cfg.get("encoder_checkpoint", None),
            decoder_checkpoint=cfg.get("decoder_checkpoint", None),
            decoder_z_scale=cfg.get("decoder_z_scale", None),
            decoder_attn_norm=cfg.get("decoder_attn_norm", None),
        )
        # Aux losses (LPIPS + REPA). REPA hooks the decoder's ViT mid-block, so build it AFTER the
        # decoder exists and BEFORE compile_submodules (forward hooks survive torch.compile).
        repa_cfg = dict(cfg.repa) if cfg.get("repa", None) is not None else None
        self.repa_cfg = repa_cfg or {}
        self.aux_losses = AuxLosses(self.net.decoder, lpips=bool(cfg.get("lpips", True)), repa=repa_cfg,
                                    image_size=self.image_size)

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compile_submodules(self) -> None:
        """torch.compile the encoder + decoder only (Lightning hooks/logging stay eager)."""
        self.net.encoder = torch.compile(self.net.encoder)
        self.net.decoder = torch.compile(self.net.decoder)

    # ===================== obs prep ======================================= #
    def _images(self, batch: dict, key: Optional[str] = None) -> torch.Tensor:
        """batch -> (B*T, 3, H, W) in [-1,1] for ONE view (uint8 -> float/255 -> [0,1]->[-1,1])."""
        k = key or self.obs_keys[0]
        x = batch["obs"][k]
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        x = self.normalizer[self.obs_keys[0]].normalize(x)   # (B,T,3,H,W) in [-1,1]
        return rearrange(x, "b t c h w -> (b t) c h w")

    # ===================== training ======================================= #
    def training_step(self, batch: dict, batch_idx: int):
        if batch["obs"][self.obs_keys[0]].shape[0] == 0:
            return None
        self._log_progress(batch["obs"][self.obs_keys[0]].shape[0], batch_idx)
        x = self._images(batch)

        out = self.net(x)
        losses = dict(out.losses)
        losses.update(self.aux_losses(x, out.x0_pred))
        u = self._latent_uniformity(out.z)     # the ONLY loss term that sees the latent
        if u is not None:
            losses["uniformity"] = u
            with torch.no_grad():
                # mrl = batch mean resultant length of the unit tokens: ~0 uniform, ->1 collapsed.
                zt = out.z.flatten(2).transpose(1, 2).reshape(-1, out.z.shape[1]).float()
                zt = zt / zt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                self.log("training/latent_mrl", zt.mean(0).norm())
                self.log("training/uniformity_coef", self._uniformity_coef())
                self.log("training/repa_coef", self._repa_coef())
        total = self._aggregate(losses)

        zs = getattr(self.net.decoder, "z_scale", None)
        if zs is not None:
            self.log("training/z_scale", zs.detach())
        self.log("training/loss", total, prog_bar=True)
        for k, v in losses.items():
            self.log(f"training/{k}", v.detach())
        return {"loss": total}

    def _latent_uniformity(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Hinged Gaussian potential ``relu(m - c)^2``, ``m = E[exp(-t||z_i-z_j||^2)]`` over
        ``n_pairs`` token pairs drawn from DIFFERENT batch elements; 0 iff the marginal is uniform."""
        if not bool(self.unif_cfg.get("enabled", False)):
            return None
        w = float(self.unif_cfg.get("t", 2.0))
        n_pairs = int(self.unif_cfg.get("n_pairs", 4096))
        B = z.shape[0]
        zt = z.flatten(2).transpose(1, 2)                        # (B, P, C)
        zt = zt / zt.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # already unit, enforce anyway
        P = zt.shape[1]
        dev = z.device
        if B > 1:
            bi = torch.randint(0, B, (n_pairs,), device=dev)
            off = torch.randint(1, B, (n_pairs,), device=dev)   # never 0 -> always a DIFFERENT element
            bj = (bi + off) % B
        else:
            bi = torch.randint(0, B, (n_pairs,), device=dev)
            bj = torch.randint(0, B, (n_pairs,), device=dev)
        pi = torch.randint(0, P, (n_pairs,), device=dev)
        pj = torch.randint(0, P, (n_pairs,), device=dev)
        a = zt[bi, pi]                                           # (n_pairs, C)
        b = zt[bj, pj]
        # ||a-b||^2 = 2 - 2<a,b> for unit vectors; keep it in fp32 (the exp is scale-sensitive).
        d2 = (2.0 - 2.0 * (a.float() * b.float()).sum(-1)).clamp_min(0)
        m = torch.exp(-w * d2).mean()
        c = float(self.unif_cfg["uniform_const"])
        return (m - c).clamp_min(0.0).square()

    def _uniformity_coef(self) -> float:
        return float(self.loss_coef.get("uniformity", 0.0))

    def _repa_coef(self) -> float:
        """loss_coef.repa, ramped linearly from 0 over ``repa.warmup_steps`` (the projector is random at init)."""
        coef = float(self.loss_coef.get("repa", 0.0))
        wu = int(self.repa_cfg.get("warmup_steps", 0) or 0)
        if wu > 0:
            coef *= min(1.0, float(self.global_step) / float(wu))
        return coef

    def _aggregate(self, losses: dict) -> torch.Tensor:
        terms = []
        for name, val in losses.items():
            if name == "uniformity":
                coef = self._uniformity_coef()
            elif name == "repa":
                coef = self._repa_coef()
            else:
                coef = self.loss_coef.get(name, 0.0)
            if coef != 0 and val is not None:
                terms.append(coef * val)
        return torch.stack(terms).sum()

    # ===================== validation ===================================== #
    def _ssim(self, rc01: torch.Tensor, gt01: torch.Tensor) -> torch.Tensor:
        m = getattr(self, "_ssim_metric", None)
        if m is None:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            m = StructuralSimilarityIndexMeasure(data_range=1.0).to(rc01.device)
            object.__setattr__(self, "_ssim_metric", m)   # not a registered submodule
        return m(rc01, gt01)

    @torch.no_grad()
    def _score_view(self, batch: dict, view: str, batch_idx: int, multi: bool) -> None:
        """Encode->decode ONE view at ``val_steps`` NFE; log PSNR/MSE/SSIM/LPIPS/mrl + stash panel frames."""
        x = self._images(batch, key=view)               # (N,3,H,W) in [-1,1]
        z = self.net.encode(x)
        recon = self.net.decode(z, steps=int(self.cfg.get("val_steps", 8)))

        gt01 = ((x + 1) / 2).clamp(0, 1)
        rc01 = ((recon.float() + 1) / 2).clamp(0, 1)
        mse = torch.mean((gt01 - rc01) ** 2, dim=[1, 2, 3])
        psnr = (-10.0 * torch.log10(mse.clamp_min(1e-10))).mean()
        p = f"validation/{view}/" if multi else "validation/"
        self.log(f"{p}psnr", psnr, sync_dist=True, prog_bar=not multi)
        self.log(f"{p}mse", mse.mean(), sync_dist=True)
        self.log(f"{p}ssim", self._ssim(rc01, gt01), sync_dist=True, prog_bar=not multi)
        if self.aux_losses.lpips_loss is not None:
            lp = self.aux_losses.lpips_loss(recon.float().clamp(-1, 1), x.clamp(-1, 1)).mean()
            self.log(f"{p}lpips", lp, sync_dist=True)
        zt = z.flatten(2).transpose(1, 2).float()
        zt = zt / zt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.log(f"{p}mrl", zt.reshape(-1, zt.shape[-1]).mean(0).norm(), sync_dist=True)

        # Reservoir-sample panel frames per view (rank 0 only), uniform over the whole val pass.
        if self.global_rank == 0:
            pool = self.validation_step_outputs.setdefault(view, [])
            seen = self._val_seen.get(view, 0)
            gen = torch.Generator().manual_seed(self.global_step * 100003 + batch_idx)
            for j in range(rc01.shape[0]):
                seen += 1
                frame = (rc01[j:j + 1].cpu(), gt01[j:j + 1].cpu())
                if len(pool) < self._panel_pool:
                    pool.append(frame)
                else:
                    r = int(torch.randint(seen, (1,), generator=gen).item())
                    if r < self._panel_pool:
                        pool[r] = frame
            self._val_seen[view] = seen

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int):
        # Per-view val: score every view present in the batch separately (top/left/right).
        views = [v for v in ("top", "left", "right") if v in batch["obs"]] or list(batch["obs"])
        multi = len(views) > 1
        for v in views:
            self._score_view(batch, v, batch_idx, multi)

    def on_validation_epoch_end(self, max_images: int = 8) -> None:
        pools, seen_map = self.validation_step_outputs, self._val_seen
        self.validation_step_outputs, self._val_seen = {}, {}
        if not pools or self.logger is None or not hasattr(self.logger, "log_image") or self.global_rank != 0:
            return
        multi = len(pools) > 1
        for view, pool in pools.items():
            if not pool:
                continue
            rc = torch.cat([r for r, _ in pool], 0)       # (K,C,H,W) reservoir
            gt = torch.cat([g for _, g in pool], 0)
            n = min(rc.shape[0], max_images)
            g = torch.Generator().manual_seed(self.global_step + hash(view) % 1000)
            idx = torch.randperm(rc.shape[0], generator=g)[:n].tolist()
            images, captions = [], []
            for i in idx:
                pair = torch.cat([rc[i], gt[i]], dim=-1)  # (C,H,2W)
                images.append((pair.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy())
                captions.append(f"{view} sample {i} of {seen_map.get(view, 0)} | left: recon, right: gt")
            key = f"validation/{view}/reconstructions" if multi else "validation/reconstructions"
            self.logger.log_image(key=key, images=images, caption=captions, step=self.global_step)

    # ===================== optimizer (ScheduleFree) ======================= #
    def configure_optimizers(self):
        from schedulefree import RAdamScheduleFree

        base_lr = float(self.cfg.base_lr)
        backbone_lr = float(self.cfg.backbone_lr)
        wd = float(self.cfg.get("weight_decay", 0.0))
        # Pretrained (SSDD-initialised) encoder + decoder at backbone_lr; everything else trainable
        # (the randomly initialised REPA projector) at base_lr. torch.compile inserts "_orig_mod."
        # into parameter names, so strip it before matching.
        backbone, fresh = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            norm = name.replace("._orig_mod.", ".")
            (backbone if norm.startswith(("net.encoder.", "net.decoder.")) else fresh).append(p)
        print(f"[lr-groups] base_lr={base_lr:g} -> {len(fresh)} tensors / "
              f"{sum(p.numel() for p in fresh):,} params   backbone_lr={backbone_lr:g} -> "
              f"{len(backbone)} tensors / {sum(p.numel() for p in backbone):,} params", flush=True)
        groups = [{"params": fresh, "lr": base_lr}]
        if backbone:
            groups.append({"params": backbone, "lr": backbone_lr})
        # ScheduleFree folds the LR schedule internally; no external scheduler.
        optimizer = RAdamScheduleFree(groups, lr=base_lr, weight_decay=wd)
        self._sf_optimizer = optimizer
        return optimizer

    # ScheduleFree swaps between the live "y" iterate (train) and the averaged "x" iterate (eval).
    # Train mode is re-asserted at every train-batch start so a mid-epoch validation cannot leave
    # the optimizer in eval; eval mode is set for validation and for checkpoint serialization.
    def _sf_set(self, train: bool) -> None:
        opt = getattr(self, "_sf_optimizer", None)
        if opt is not None and hasattr(opt, "train"):
            opt.train() if train else opt.eval()

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self._sf_set(True)

    def on_validation_epoch_start(self) -> None:
        self._sf_set(False)

    def state_dict(self, *args, **kwargs):
        """Serialize the ScheduleFree EVAL (averaged) iterate ``x``, never the training iterate ``y``.
        Lightning fills ``state_dict`` BEFORE ``optimizer_states``/``on_save_checkpoint``, so this is
        the only hook that reaches the saved weights; the saved ``train_mode=False`` then matches them."""
        self._sf_set(False)
        return super().state_dict(*args, **kwargs)

    def on_train_start(self) -> None:
        self._t_start = time.time()
        self._samples_seen = 0

    # ===================== progress / gradient health ===================== #
    def _log_progress(self, batch_size: int, batch_idx: int) -> None:
        if self._t_start is None:
            self._t_start = time.time()
        self._samples_seen += batch_size
        if batch_idx % 50 == 0:
            elapsed = max(time.time() - self._t_start, 1e-9)
            self.log("training/samples_per_s", self._samples_seen / elapsed, prog_bar=False)
            self.log("progress/epoch", float(self.current_epoch), on_step=True, rank_zero_only=True)
            if self.steps_per_epoch_estimate:
                epoch_pct = 100.0 * (batch_idx + 1) / self.steps_per_epoch_estimate
                self.log("progress/epoch_percent", min(epoch_pct, 100.0), on_step=True, rank_zero_only=True)

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log the global PRE-clip gradient norm and ABORT on a non-finite one (before clipping can
        propagate NaN into the weights or a NaN checkpoint gets written)."""
        norms = [p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None]
        if not norms:
            return
        total = torch.linalg.vector_norm(torch.stack([n.float() for n in norms]))
        if not bool(torch.isfinite(total)):
            raise RuntimeError(
                f"non-finite pre-clip gradient norm ({total.item()}) at global_step {self.global_step} "
                f"(rank {self.global_rank}): the run has diverged. Resume from an earlier checkpoint.")
        if self.logger is not None:
            self.log("training/grad_norm", total, prog_bar=False, rank_zero_only=True)
