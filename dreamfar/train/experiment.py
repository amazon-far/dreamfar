"""Lightning wiring shared by the tokenizer and dynamics trainers: data, DDP, checkpoints, resume."""
from __future__ import annotations

import datetime
import json
import os

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_comm_hooks

torch.set_float32_matmul_precision("high")


def _ddp_strategy(ecfg: DictConfig) -> DDPStrategy:
    """DDP with bf16 gradient all-reduce (halves the wire bytes; training is bf16-mixed anyway)."""
    return DDPStrategy(
        find_unused_parameters=bool(ecfg.ddp.find_unused_parameters),
        gradient_as_bucket_view=True,
        ddp_comm_hook=ddp_comm_hooks.bf16_compress_hook if ecfg.ddp.comm_hook == "bf16" else None,
        timeout=datetime.timedelta(minutes=int(ecfg.ddp_timeout_min)),
    )


class Experiment:
    def __init__(self, cfg: DictConfig, logger=None):
        self.cfg = cfg
        self.ecfg = cfg.experiment
        self.logger = logger
        self.kind = str(cfg.algorithm.kind)                  # tokenizer | dynamics

    # -- data ------------------------------------------------------------- #
    def _tokenizer_data(self):
        from dreamfar.tokenizer.data import FrameDataset, make_frame_loader, split_episodes
        d = self.cfg.dataset
        index = json.load(open(os.path.join(d.raw_root, "index.json")))
        train_ids, val_ids = split_episodes(index, heldout_path=d.heldout_path, val_ratio=d.val_ratio,
                                            seed=d.seed)
        val_ids = val_ids[:int(d.val_max_episodes)] if d.val_max_episodes else val_ids
        views = tuple(d.views)
        train_ds = FrameDataset(d.raw_root, train_ids, resolution=int(d.resolution), views=views,
                                random_view=True, hflip_prob=float(d.hflip_prob), frame_stride=int(d.frame_stride))
        val_ds = FrameDataset(d.raw_root, val_ids, resolution=int(d.resolution), views=views,
                              random_view=False, hflip_prob=0.0, all_views=True, frame_stride=int(d.val_frame_stride))
        t, v = self.ecfg.training, self.ecfg.validation
        return (make_frame_loader(train_ds, t.batch_size, int(d.num_workers), shuffle=True),
                make_frame_loader(val_ds, v.batch_size, int(d.val_num_workers), shuffle=True))

    def _dynamics_data(self, model):
        from dreamfar.data.latent_dataset import LatentShardDataset, load_catalog, make_latent_loader, split_heldout
        from dreamfar.data.normalizer import load_normalizer_file
        d = self.cfg.dataset
        episodes, meta = load_catalog(d.latent_root)
        train_eps, val_eps = split_heldout(episodes, d.heldout_path)
        model.set_normalizer(load_normalizer_file(d.normalizer_path))
        common = dict(latent_shape=tuple(meta["latent_shape"]), latent_dtype=meta["latent_dtype"],
                      frame_skip=int(d.frame_skip), contact_min_dwell=int(d.contact_min_dwell), fps=float(meta["fps"]))
        train_ds = LatentShardDataset(
            train_eps, sequence_length=int(d.horizon), contact_labels_dir=d.contact_dir,
            window_stride=int(d.window_stride), random_start_jitter=True, n_memory=int(d.n_memory),
            n_mem_pool=int(d.n_mem_pool), mem_dropout_prob=float(d.mem_dropout_prob), stochastic_select=True,
            kf_sim_weight=float(d.kf_sim_weight), kf_time_weight=float(d.kf_time_weight),
            kf_normalize_sim=bool(d.kf_normalize_sim), kf_dt_ref_s=float(d.kf_dt_ref_s),
            mcp_target_frames=int(d.mcp_target_frames), **common)
        val_ds = LatentShardDataset(val_eps, sequence_length=int(d.val_horizon), contact_labels_dir=d.contact_dir,
                                    window_stride=1, random_start_jitter=False, **common)
        print(f"[data] {len(train_ds.episodes)} train / {len(val_ds.episodes)} val episodes, "
              f"{len(train_ds)} train windows", flush=True)
        t, v = self.ecfg.training, self.ecfg.validation
        return (make_latent_loader(train_ds, t.batch_size, int(d.num_workers), shuffle=True, seed=int(d.seed)),
                make_latent_loader(val_ds, v.batch_size, int(d.val_num_workers), shuffle=False, seed=int(d.seed)))

    # -- training --------------------------------------------------------- #
    def train(self) -> None:
        if self.kind == "tokenizer":
            from dreamfar.tokenizer.module import TokenizerModule
            model = TokenizerModule(self.cfg.algorithm)
            train_loader, val_loader = self._tokenizer_data()
        else:
            from dreamfar.dynamics.module import DynamicsModule
            model = DynamicsModule(self.cfg.algorithm)
            train_loader, val_loader = self._dynamics_data(model)
        t, v = self.ecfg.training, self.ecfg.validation
        if bool(t.compile):
            model.compile_submodules()
        ckpt_dir = os.path.join(str(self.cfg.ckpt_root), str(self.cfg.name))
        callbacks = [ModelCheckpoint(ckpt_dir, every_n_train_steps=int(t.checkpointing.every_n_train_steps),
                                     save_last=True, save_top_k=1, monitor=None, enable_version_counter=False)]
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        world = int(self.ecfg.num_devices) * int(self.ecfg.num_nodes)
        print(f"[train] {self.kind}: batch {int(t.batch_size)} x world {world} x accum "
              f"{int(t.accumulate_grad_batches)} = {int(t.batch_size) * world * int(t.accumulate_grad_batches)} "
              f"samples per optimizer step; checkpoints -> {ckpt_dir}", flush=True)
        trainer = pl.Trainer(
            accelerator="auto", devices=int(self.ecfg.num_devices), num_nodes=int(self.ecfg.num_nodes),
            strategy=(_ddp_strategy(self.ecfg) if world > 1 else "auto"),
            logger=self.logger if self.logger else False, callbacks=callbacks,
            precision=str(t.precision), max_steps=int(t.max_steps), max_epochs=-1,
            gradient_clip_val=float(t.gradient_clip_val), accumulate_grad_batches=int(t.accumulate_grad_batches),
            val_check_interval=int(v.val_every_n_step), check_val_every_n_epoch=None,
            limit_val_batches=int(v.limit_batch), num_sanity_val_steps=0,
            log_every_n_steps=int(t.log_every_n_steps),
        )
        resume = self.cfg.resume
        if resume in (None, "", "auto"):
            last = os.path.join(ckpt_dir, "last.ckpt")
            resume = last if os.path.exists(last) else None
            print(f"[resume] {'resuming from ' + last if resume else 'starting from scratch'}", flush=True)
        elif resume == "scratch":
            resume = None
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume)
