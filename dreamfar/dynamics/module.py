"""Streaming latent dynamics model (DreamFAR stage 2): training loss, EMA, validation and the
KV-cached streaming rollout, on latents of a frozen tokenizer.

Sequence layout: ``[sink(1) | memory(M) | context c(T)]``. Training = iid diffusion forcing over the T
context frames (flow-matching velocity loss on positions 1..T-1), memory at a fixed flow-time, the
state token denoised jointly, contact read out of a middle block, MCP auxiliary heads. Inference =
P-deep pyramid with one committed frame per step, un-roped KV cache for sink/memory/committed frames.
"""
from __future__ import annotations

import math
import random
import time
from collections import OrderedDict
from typing import Callable, Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from dreamfar.data.contact import ContactKeyframeTracker
from dreamfar.data.ee_fk import EeFk
from dreamfar.data.latent_dataset import keyframe_confidence
from dreamfar.data.normalizer import LinearNormalizer
from dreamfar.dynamics.dit import DiTLatentDynamics
from dreamfar.dynamics.optim import split_muon_adamw_params
from dreamfar.dynamics.stream_layout import (build_stream_attn_mask, build_stream_cond_mask,
                                             build_stream_rope_pos, dt_to_rope_pos)
from dreamfar.dynamics.wrappers import EinopsWrapper
from dreamfar.tokenizer.net import load_tokenizer


def _rope_shift(rope_pos: torch.Tensor, c0: int, k: int) -> torch.Tensor:
    """Advance the context rows' rotary positions by k frames (MCP depth k predicts frame i+k)."""
    out = rope_pos.clone()
    out[..., c0:] = out[..., c0:] + float(k)
    return out


def _assemble_ctx(sink_cache, mem_kvs, cmt_cache, nsk, M, n_c, dev, mem_pos, pad_to=None):
    """Concatenate the per-block [sink | memory | committed] un-roped KV into one context cache.

    ``pad_to`` pads the committed block to a fixed width (static shape for torch.compile); the padded
    columns repeat the last committed frame and must be masked off by the caller."""
    nb = (len(sink_cache) if sink_cache is not None
          else len(mem_kvs[0]) if mem_kvs else len(cmt_cache[0]))
    n_cc = n_c if pad_to is None else pad_to
    pad = 0 if pad_to is None else pad_to - n_c
    ctx_k, ctx_v = [], []
    for bi in range(nb):
        ks, vs = [], []
        if nsk:
            ks.append(sink_cache[bi][0]); vs.append(sink_cache[bi][1])
        for mk in mem_kvs:
            ks.append(mk[bi][0]); vs.append(mk[bi][1])
        for ck in cmt_cache:
            ks.append(ck[bi][0]); vs.append(ck[bi][1])
        if pad > 0:
            # `repeat`, not `expand`: a joint block stores all tokens of a frame per column group.
            fk, fv = ks[-1], vs[-1]
            ks.append(fk.repeat(1, 1, pad, 1))
            vs.append(fv.repeat(1, 1, pad, 1))
        ctx_k.append(torch.cat(ks, dim=2)); ctx_v.append(torch.cat(vs, dim=2))
    pos = [torch.zeros(nsk, device=dev)]
    if M:
        pos.append(mem_pos.to(dev))
    pos.append(torch.arange(n_cc, device=dev, dtype=torch.float32))
    return ctx_k, ctx_v, torch.cat(pos), nsk + M + n_cc


def _snapshot_committed(dit, x_i, st_i, act, i, total, s_lvl, ts, ad, nrm, sn, cn,
                        ctx_k, ctx_v, ctx_pos, n_ctx, pos_i, dev, B, n_real=None, decode_fn=None):
    """Snapshot committed frame ``i``'s per-block un-roped KV at flow-time ``s_lvl`` against the given
    [sink|mem|committed] context (a seq-len-1 decode). ``n_real`` masks off padded ctx columns."""
    ci_x = ((1 - s_lvl) * x_i + s_lvl * cn[i]).unsqueeze(2)                       # (B,C,1,h,w)
    ci_s = ((1 - s_lvl) * nrm.normalize(st_i) + s_lvl * sn[i]).unsqueeze(0)       # (1,B,pose)
    a = min(total - 1, max(0, i - 1))
    cond = torch.zeros_like(act[a:a + 1]) if i == 0 else act[a:a + 1].clone()     # (1,B,A)
    wdt = x_i.dtype
    pdt = next(dit.parameters()).dtype
    mask = torch.ones((B, 1, n_ctx + 1), dtype=torch.bool, device=dev)
    if n_real is not None and n_real < n_ctx:
        mask[:, :, n_real:n_ctx] = False
    _, _, _, kv = (decode_fn or dit.stream_decode)(
        ci_x.to(pdt), torch.full((1, B), s_lvl * ts, device=dev, dtype=wdt), cond.to(pdt), ci_s.to(pdt),
        [(ck, cv) for ck, cv in zip(ctx_k, ctx_v)], ctx_pos,
        torch.tensor([float(pos_i)], device=dev), mask,
        cond_mask=torch.ones((B, 1), device=dev, dtype=pdt))
    return kv


class DynamicsModule(pl.LightningModule):
    """Flow-matching streaming latent dynamics on a frozen tokenizer."""

    VAL_FLOW_BUCKETS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.obs_keys = list(cfg.obs_keys)
        self.num_views = len(self.obs_keys)
        self.image_size = int(cfg.image_size)
        self.z_dim = int(cfg.z_dim)
        self.n_tokens = int(cfg.n_frames)                      # window length T
        self.flow_timescale = float(cfg.flow_timescale)
        self.dyn_flow_steps = int(cfg.dyn_flow_steps)          # pyramid depth P
        self.logit_normal_mean = float(cfg.logit_normal_mean)
        self.logit_normal_std = float(cfg.logit_normal_std)
        self.infer_t_schedule = str(cfg.infer_t_schedule)     # uniform | logit_normal
        if self.infer_t_schedule not in ("uniform", "logit_normal"):
            raise ValueError(f"infer_t_schedule must be uniform|logit_normal, got {self.infer_t_schedule!r}")
        self.infer_weights_dtype = str(cfg.infer_weights_dtype)   # fp32 | bf16
        self.stabilization_level = float(cfg.stabilization_level)
        self.stream_snap_reuse = bool(cfg.stream_snap_reuse)
        _ctxn = cfg.stream_ctx_noise
        if _ctxn is None:
            self.stream_ctx_noise = None
        elif isinstance(_ctxn, (list, tuple)) or (hasattr(_ctxn, "__iter__") and not isinstance(_ctxn, str)):
            self.stream_ctx_noise = [float(x) for x in _ctxn]
        else:
            self.stream_ctx_noise = float(_ctxn)
        self.n_memory = int(cfg.n_memory)
        self.kf_sim_weight = float(cfg.kf_sim_weight)
        self.kf_time_weight = float(cfg.kf_time_weight)
        self.kf_normalize_sim = bool(cfg.kf_normalize_sim)
        _dt = float(cfg.kf_dt_ref_s)
        self.kf_dt_ref = _dt * 30.0 if _dt > 0 else None      # native (30 Hz) frames
        self.ee_urdf_path = str(cfg.ee_urdf_path or "")
        self._ee_fk = EeFk(self.ee_urdf_path) if self.ee_urdf_path else None
        self.contact_readout_layer = int(cfg.contact_readout_layer)
        self.contact_infer_t = float(cfg.contact_infer_t)
        self.contact_min_dwell = int(cfg.contact_min_dwell)
        self.contact_pred_weight = float(cfg.contact_pred_weight)
        self.contact_dim = 2
        self.stream_cwl = bool(cfg.stream_cwl)
        self.stream_cwl_w_none = float(cfg.stream_cwl_w_none)
        self.stream_cwl_w_contact = float(cfg.stream_cwl_w_contact)
        self.stream_cwl_w_near = float(cfg.stream_cwl_w_near)
        self.stream_cwl_near_frames = int(cfg.stream_cwl_near_frames)
        self.stream_cwl_normalize = bool(cfg.stream_cwl_normalize)
        self.mcp_snr_shift = float(cfg.mcp_snr_shift)
        self.mcp_loss_weights = list(cfg.mcp_loss_weights)
        self.mcp_state_loss = bool(cfg.mcp_state_loss)
        self.pose_dim = int(cfg.pose_dim)                      # state token width (joints)
        self._gripper_dims = (self.pose_dim // 2 - 1, self.pose_dim - 1)
        self._action_dim = int(cfg.action_dim)
        self.state_pred_weight = float(cfg.state_pred_weight)
        self.frame_skip = int(cfg.frame_skip)
        self.stream_dt_span = float(cfg.streaming.dt_span)
        self.stream_dt_ref = float(cfg.streaming.dt_ref)
        self.stream_kv_pool_resident = int(cfg.streaming.kv_pool_resident)
        self.clip_noise = float(cfg.clip_noise)
        ema = cfg.ema_decay
        self.ema_decay = float(ema) if ema not in (None, 0, 0.0) else 0.0
        self.ema_warmup = bool(cfg.ema_warmup)
        self._last_ema_decay = self.ema_decay
        self._ema_state: Optional[dict] = None
        self._ema_backup: Optional[dict] = None
        self.val_stream_ar_batches = int(cfg.val_stream_ar_batches)
        self.val_ar_rows = int(cfg.val_ar_rows)
        self.val_sample_seed = int(cfg.val_sample_seed)
        self._ar_ids: Optional[set] = None
        self._t_start: Optional[float] = None
        self._samples_seen = 0
        self._build_model()
        self.normalizer = LinearNormalizer()

    # ===================== model construction ============================= #
    def _build_model(self) -> None:
        cfg = self.cfg
        # Frozen tokenizer: decoder only (latents are precomputed; the encoder stub keeps patch_size).
        self.tokenizer = load_tokenizer(cfg.tokenizer_ckpt, cfg.tokenizer_config, drop_encoder=True)
        for p in self.tokenizer.parameters():
            p.requires_grad = False
        self.tokenizer.eval()
        assert int(self.tokenizer.z_dim) == self.z_dim, \
            f"tokenizer latent dim {self.tokenizer.z_dim} != z_dim {self.z_dim}"
        lat = self.image_size // int(self.tokenizer.encoder.patch_size)
        dit = OmegaConf.to_container(cfg.dit, resolve=True)
        dyn = DiTLatentDynamics(
            latent_dim=self.z_dim, action_dim=self._action_dim, state_dim=self.pose_dim,
            spatial_grid=[self.num_views * lat, lat],                # views stacked on H
            contact_dim=self.contact_dim, contact_readout_layer=self.contact_readout_layer,
            grad_checkpoint=bool(cfg.grad_checkpoint), mlp_hidden_multiple=int(cfg.mlp_hidden_multiple),
            **dit,
        )
        self.dynamics = EinopsWrapper(from_shape="f b c h w", to_shape="b c f h w", module=dyn)
        self._cm = self.dynamics.module                        # the DiT (owns stream_prefill/decode)
        init_ckpt = cfg.init_dynamics_ckpt
        if init_ckpt:
            # Weights-only warm start (fresh optimizer / step).
            ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            sd = {k.replace("._orig_mod.", "."): v for k, v in ck["state_dict"].items()}
            dyn_sd = {k[len("dynamics."):]: v for k, v in sd.items() if k.startswith("dynamics.")}
            missing, unexpected = self.dynamics.load_state_dict(dyn_sd, strict=False)
            if self._cm.mcp is not None and not any(".mcp." in k for k in dyn_sd):
                self._cm.init_mcp_from_backbone()
            print(f"[init-dynamics] warm start from {init_ckpt} (step {ck.get('global_step', '?')}): "
                  f"{len(dyn_sd)} tensors, missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compile_submodules(self) -> None:
        """torch.compile each DiT block (keeps activation checkpointing eager and effective)."""
        blocks = self._cm.blocks
        for i in range(len(blocks)):
            blocks[i] = torch.compile(blocks[i])
        n_mcp = 0
        if self._cm.mcp is not None:
            for g in self._cm.mcp.blocks:
                for i in range(len(g)):
                    g[i] = torch.compile(g[i]); n_mcp += 1
        print(f"[compile] per-block: {len(blocks)} DiT blocks + {n_mcp} MCP blocks", flush=True)

    # ===================== latent / cond helpers ========================== #
    def stack_views(self, lat: torch.Tensor) -> torch.Tensor:
        """Per-view latents (B,T,V,C,h,w) -> view-stacked (T,B,C,V*h,w) float32."""
        assert lat.shape[2] == self.num_views, f"latent V={lat.shape[2]} != num_views={self.num_views}"
        if self.num_views > 1:
            lat = rearrange(lat, "b t v c h w -> t b c (v h) w")
        else:
            lat = rearrange(lat[:, :, 0], "b t c h w -> t b c h w")
        return lat.float()

    def split_views(self, z: torch.Tensor) -> list:
        """(..., C, V*h, w) -> list of V per-view latents (..., C, h, w)."""
        if self.num_views == 1:
            return [z]
        return list(torch.chunk(z, self.num_views, dim=-2))

    def cond_action(self, action: torch.Tensor) -> torch.Tensor:
        """Raw EE action -> the normalized conditioning slot (`cond_action_norm=own`)."""
        return self.normalizer["action"].normalize(action.float()).float()

    def apply_infer_weights_dtype(self) -> torch.dtype:
        """Cast the dynamics weights to `infer_weights_dtype` (standalone inference only)."""
        if self.infer_weights_dtype == "bf16":
            self.dynamics.to(torch.bfloat16)
        return next(self.dynamics.parameters()).dtype

    def _renorm_per_view(self, frames: torch.Tensor) -> torch.Tensor:
        """Re-project each view of (K,B,C,V*h,w) onto its own unit sphere."""
        if self.num_views == 1:
            return frames / (torch.norm(frames, dim=2, keepdim=True) + 1e-8)
        vs = torch.chunk(frames, self.num_views, dim=3)
        vs = [v / (torch.norm(v, dim=2, keepdim=True) + 1e-8) for v in vs]
        return torch.cat(vs, dim=3)

    def _infer_t_levels(self, P: int):
        """Pyramid flow-time grid: (levels t_0 > ... > t_P = 0, Euler steps dts[k] = t_k - t_{k+1})."""
        u = [1.0 - k / P for k in range(P + 1)]
        if self.infer_t_schedule == "uniform":
            return u, [1.0 / P] * P
        from statistics import NormalDist
        nd = NormalDist(self.logit_normal_mean, max(self.logit_normal_std, 1e-8))

        def q(p):                                   # quantile of the training law sigmoid(N(mu, sd))
            return 1.0 / (1.0 + math.exp(-nd.inv_cdf(p)))
        lv = [1.0] + [q(t) for t in u[1:P]] + [0.0]
        lv[0], lv[-1] = float(lv[0]), 0.0
        return lv, [lv[k] - lv[k + 1] for k in range(P)]

    def _contact_readout_slot(self, levels) -> int:
        """In-flight slot whose query flow-time is nearest `contact_infer_t`."""
        P = len(levels) - 1
        return int(np.argmin([abs(levels[P - 1 - j] - self.contact_infer_t) for j in range(P)]))

    def _sample_t(self, shape, device) -> torch.Tensor:
        """Logit-normal flow-time draw sigmoid(randn*std + mean)."""
        return torch.sigmoid(torch.randn(shape, device=device) * self.logit_normal_std
                             + self.logit_normal_mean)

    def _ctx_noise_weights(self, levels: list) -> list:
        """Training-frequency weights of discrete context flow-times (nearest-value bins under the
        logit-normal training law, symmetric half-spacing edges)."""
        lv = sorted(float(x) for x in levels)
        half = (lv[1] - lv[0]) / 2 if len(lv) > 1 else lv[0] / 2
        mids = [(lv[i] + lv[i + 1]) / 2 for i in range(len(lv) - 1)]
        edges = [max(0.0, lv[0] - half)] + mids + [min(1.0, lv[-1] + half)]

        def _cdf(t):
            if t <= 0.0:
                return 0.0
            if t >= 1.0:
                return 1.0
            z = (math.log(t / (1 - t)) - self.logit_normal_mean) / max(self.logit_normal_std, 1e-8)
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        m = [_cdf(edges[i + 1]) - _cdf(edges[i]) for i in range(len(lv))]
        tot = sum(m)
        return [x / tot for x in m] if tot > 0 else [1.0 / len(lv)] * len(lv)

    def _sample_t_mcp(self, shape, device) -> torch.Tensor:
        """MCP flow-time: the main draw shifted toward high noise, t' = s*t / (1 + (s-1)*t)."""
        t = self._sample_t(shape, device)
        s = self.mcp_snr_shift
        return t if s == 1.0 else s * t / (1.0 + (s - 1.0) * t)

    def _contact_frame_weights(self, contact_gt: torch.Tensor, dtype) -> torch.Tensor:
        """Per-frame loss weight (T-1,B): near a contact transition > in contact > none; mean-normalized."""
        c = contact_gt > 0.5                                            # (T,B,2)
        trans = torch.zeros(c.shape[:2], dtype=torch.bool, device=c.device)
        trans[1:] = (c[1:] != c[:-1]).any(-1)
        near = trans
        w_ = self.stream_cwl_near_frames
        if w_ > 0:
            pooled = F.max_pool1d(trans.float().transpose(0, 1).unsqueeze(1),
                                  kernel_size=2 * w_ + 1, stride=1, padding=w_)
            near = pooled.squeeze(1).transpose(0, 1) > 0
        w = torch.full(c.shape[:2], self.stream_cwl_w_none, device=c.device, dtype=dtype)
        w = torch.where(c.any(-1), torch.full_like(w, self.stream_cwl_w_contact), w)
        w = torch.where(near, torch.full_like(w, self.stream_cwl_w_near), w)
        w = w[1:]
        with torch.no_grad():
            n = float(w.numel())
            self._last_scwl = {
                "near": (w == self.stream_cwl_w_near).float().sum() / n,
                "contact": (w == self.stream_cwl_w_contact).float().sum() / n,
                "none": (w == self.stream_cwl_w_none).float().sum() / n,
                "mean_w": w.mean(),
            }
        if self.stream_cwl_normalize:
            w = w / w.mean().clamp_min(1e-8)
        return w

    # ===================== training loss ================================== #
    def _flow_loss_stream(self, z: torch.Tensor, action: torch.Tensor, target_state: torch.Tensor,
                          z_mem: torch.Tensor, mem_state_norm: torch.Tensor, mem_dt: torch.Tensor,
                          mem_valid: torch.Tensor, mem_frame_mask: Optional[torch.Tensor] = None,
                          contact_gt: Optional[torch.Tensor] = None, fixed_t: Optional[float] = None,
                          gen: Optional[torch.Generator] = None, fut: Optional[dict] = None) -> torch.Tensor:
        """Sequence [sink | memory(M) | c(T)]; velocity loss on c positions 1..T-1.

        z:(T,B,C,h,w); action:(T,B,A) normalized (rolled +1 -> frame k conditions on action[k-1]);
        target_state:(T,B,pose) raw; z_mem:(M,B,C,h,w); mem_state_norm:(M,B,pose); mem_dt:(M,B) model
        frames from c[0]; mem_valid:(M,B) bool; mem_frame_mask:(B,T,M) per-frame memory visibility.
        fixed_t + gen: the deterministic validation variant. fut: MCP target frames past the window."""
        T, B = z.shape[:2]
        M = z_mem.shape[0]
        ns = 1
        dev = z.device

        def _rn(like):
            x = (torch.randn(like.shape, generator=gen, device=dev, dtype=like.dtype)
                 if gen is not None else torch.randn_like(like))
            return torch.clamp(x, -self.clip_noise, self.clip_noise)

        if fixed_t is None:
            t = self._sample_t((T, B), dev)
        else:
            t = torch.full((T, B), float(fixed_t), device=dev)
            t[0] = self.stabilization_level
        noise = _rn(z)
        t_b = t.view(T, B, 1, 1, 1)
        x_t = (1.0 - t_b) * z + t_b * noise
        target_v = z - noise
        st1 = self.normalizer["state"].normalize(target_state)
        st_noise = _rn(st1)
        t_s = t.view(T, B, 1)
        st_x = (1.0 - t_s) * st1 + t_s * st_noise
        st_target_v = st1 - st_noise
        mem_t = torch.full((M, B), self.stabilization_level, device=dev, dtype=t.dtype)
        z_mem_x = ((1.0 - mem_t.view(M, B, 1, 1, 1)) * z_mem
                   + mem_t.view(M, B, 1, 1, 1) * _rn(z_mem))
        mem_st_x = ((1.0 - mem_t.view(M, B, 1)) * mem_state_norm
                    + mem_t.view(M, B, 1) * _rn(mem_state_norm))

        def _pre(x, like):                                                  # sink placeholder block
            return torch.zeros((ns,) + tuple(like.shape[1:]), device=dev, dtype=x.dtype)
        x_full = torch.cat([_pre(z_mem_x, z_mem_x), z_mem_x, x_t], 0)        # (N,B,C,h,w)
        st_full = torch.cat([_pre(mem_st_x, mem_st_x), mem_st_x, st_x], 0)   # (N,B,pose)
        t_full = torch.cat([torch.zeros((ns, B), device=dev, dtype=t.dtype), mem_t, t], 0)
        scaled = (t_full * self.flow_timescale).to(z.dtype)
        cond_c = torch.roll(action, shifts=1, dims=0)                       # (T,B,A) = action[k-1]
        cond_full = torch.cat([torch.zeros((ns + M, B, self._action_dim), device=dev,
                                           dtype=cond_c.dtype), cond_c], 0)
        cmask = build_stream_cond_mask(M, T, n_sink=ns, device=dev,
                                      dtype=cond_c.dtype).unsqueeze(0).expand(B, -1)
        _mv = (mem_frame_mask if mem_frame_mask is not None else mem_valid.transpose(0, 1))
        attn_mask = build_stream_attn_mask(M, T, mem_valid=_mv, n_sink=ns, device=dev)
        rope_pos = build_stream_rope_pos(M, T, mem_dt.transpose(0, 1), n_sink=ns,
                                        dt_span=self.stream_dt_span, dt_ref=self.stream_dt_ref)
        # MCP: per depth k, the same sequence with the c rows carrying frame i+k at a noisier t and the
        # c RoPE positions advanced by k (training only).
        mcp_depths = self._cm.mcp_depths
        mcp_inputs = mcp_meta = None
        if mcp_depths > 0 and self.training and fixed_t is None:
            assert fut is not None, "MCP needs the loader's future target frames (mcp_target_frames)"
            mcp_inputs, mcp_meta = [], []
            z_ext = torch.cat([z, fut["z"]], 0)                               # (T+K,B,C,h,w)
            st_ext = torch.cat([st1, self.normalizer["state"].normalize(fut["state"])], 0)
            c_ext = torch.cat([cond_c, fut["action"]], 0)
            for k in range(1, mcp_depths + 1):
                zk, stk, ck = z_ext[k:k + T], st_ext[k:k + T], c_ext[k:k + T]
                idx = torch.arange(T, device=dev) + k
                fv = fut["valid"]                                             # (K,B)
                ok = torch.where((idx < T)[:, None],
                                 torch.ones((T, B), dtype=torch.bool, device=dev),
                                 fv[torch.clamp(idx - T, 0, fv.shape[0] - 1)])   # (T,B)
                tk = self._sample_t_mcp((T, B), dev)
                nzk, nsk_ = _rn(zk), _rn(stk)
                tkb, tks = tk.view(T, B, 1, 1, 1), tk.view(T, B, 1)
                t_full_k = torch.cat([torch.zeros((ns, B), device=dev, dtype=t.dtype), mem_t, tk], 0)
                mcp_inputs.append({
                    "x": (1.0 - tkb) * zk + tkb * nzk,
                    "state": (1.0 - tks) * stk + tks * nsk_,
                    "noise_levels": (t_full_k * self.flow_timescale).to(z.dtype),
                    "cond": torch.cat([torch.zeros((ns + M, B, self._action_dim), device=dev,
                                                   dtype=ck.dtype), ck], 0),
                    "rope_pos": _rope_shift(rope_pos, ns + M, k),
                })
                mcp_meta.append({"target_v": (zk - nzk).detach(), "state_v": (stk - nsk_).detach(),
                                 "valid": ok})
        dyn_out = self.dynamics(x_full, scaled, scaled, external_cond=cond_full, state=st_full,
                                rope_pos=rope_pos, cond_mask=cmask, attn_mask=attn_mask,
                                n_mem=M, n_sink=ns, mcp_inputs=mcp_inputs)
        mcp_out = None
        if mcp_inputs is not None:
            dyn_out, mcp_out = dyn_out[:-1], dyn_out[-1]
        v_full, vs_full, contact_out = dyn_out
        sup = slice(ns + M + 1, None)                     # supervised c positions 1..T-1
        err_pf = ((v_full[sup] - target_v[1:].detach()) ** 2).mean(dim=(2, 3, 4))   # (T-1,B)
        self._last_scwl = None
        if self.stream_cwl and contact_gt is not None and fixed_t is None:
            w_pf = self._contact_frame_weights(contact_gt, err_pf.dtype)
            flow = (err_pf * w_pf).sum() / w_pf.sum().clamp_min(1e-8)
        else:
            flow = err_pf.mean()
        self._last_mcp_losses = None
        if mcp_out is not None:
            self._last_mcp_losses = []
            for (v_i, v_s), meta in zip(mcp_out, mcp_meta):
                ok = meta["valid"][1:]                                             # (T-1,B)
                m = ok.to(v_i.dtype)[:, :, None, None, None]
                e = ((v_i[sup] - meta["target_v"][1:]) ** 2 * m).sum() / (
                    m.expand_as(v_i[sup]).sum().clamp_min(1.0))
                if self.mcp_state_loss:
                    ms = ok.to(v_s.dtype)[:, :, None]
                    es = ((v_s[sup] - meta["state_v"][1:]) ** 2 * ms).sum() / (
                        ms.expand_as(v_s[sup]).sum().clamp_min(1.0))
                    e = e + self.state_pred_weight * es
                self._last_mcp_losses.append(e)
        self._last_contact_bce = None
        self._last_contact_acc = None
        if contact_gt is not None:
            logit = contact_out[sup]                                        # (T-1,B,2)
            gt = contact_gt[1:].to(logit.dtype)
            self._last_contact_bce = F.binary_cross_entropy_with_logits(logit, gt)
            with torch.no_grad():
                self._contact_diag(logit > 0, gt > 0.5)
        self._last_state_loss = F.mse_loss(vs_full[sup], st_target_v[1:].detach())
        with torch.no_grad():
            s_hat = st_x[1:] + t_s[1:] * vs_full[sup]
            sp = self.normalizer["state"].unnormalize(s_hat)
            jmask = torch.ones(self.pose_dim, device=sp.device, dtype=torch.bool)
            for gi in self._gripper_dims:
                jmask[gi] = False
            self._last_state_mae_rad = (sp - target_state[1:]).abs()[..., jmask].mean()
            z_hat = x_t[1:] + t_b[1:] * v_full[sup]
            self._last_x0_mse = F.mse_loss(z_hat, z[1:])
            self._last_x0_cos = F.cosine_similarity(z_hat, z[1:], dim=2).mean()
            err = err_pf.detach()
            ts = t[1:]
            self._last_flow_by_t = {}
            for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
                m = (ts >= lo) & (ts < hi if hi < 1.0 else ts <= 1.0)
                if m.any():
                    self._last_flow_by_t[f"{lo:g}_{hi:g}"] = err[m].mean()
            self._last_mem_valid_frac = mem_valid.float().mean()
        return flow

    def _contact_diag(self, pos: torch.Tensor, tgt: torch.Tensor) -> None:
        self._last_contact_acc = (pos == tgt).float().mean()
        self._last_contact_acc_l = (pos[..., 0] == tgt[..., 0]).float().mean()
        self._last_contact_acc_r = (pos[..., 1] == tgt[..., 1]).float().mean()
        p = tgt.float().mean()
        self._last_contact_chance = torch.maximum(p, 1.0 - p)
        self._last_contact_persist = ((tgt[1:] == tgt[:-1]).float().mean()
                                      if tgt.shape[0] >= 2 else None)

    def training_step(self, batch: dict, batch_idx: int):
        B = batch["latent"].shape[0]
        if B == 0:
            return None
        self._log_progress(B, batch_idx)
        z = self.stack_views(batch["latent"])                                   # (T,B,C,V*h,w)
        action = rearrange(self.cond_action(batch["action"]), "b t a -> t b a")
        state = rearrange(batch["state"].float(), "b t a -> t b a")             # raw joints
        z_mem = self.stack_views(batch["latent_memory"])                        # (M,B,C,h,w)
        mem_state = rearrange(batch["state_memory"].float(), "b m a -> m b a")
        mem_dt = rearrange(batch["mem_dt"].float(), "b m -> m b")
        mem_valid = rearrange(batch["mem_valid"], "b m -> m b")
        contact_gt = rearrange(batch["contact"].float(), "b t k -> t b k") if "contact" in batch else None
        fut = None
        if "latent_future" in batch:
            fut = {"z": self.stack_views(batch["latent_future"]),
                   "state": rearrange(batch["state_future"].float(), "b k a -> k b a"),
                   "action": rearrange(batch["action_future"].float(), "b k a -> k b a"),
                   "valid": rearrange(batch["future_valid"], "b k -> k b")}
        flow = self._flow_loss_stream(z, action, state, z_mem, self.normalizer["state"].normalize(mem_state),
                                      mem_dt, mem_valid, mem_frame_mask=batch.get("mem_frame_mask"),
                                      contact_gt=contact_gt, fut=fut)
        self.log("training/flow_loss", flow, prog_bar=True)
        self.log("training/state_loss", self._last_state_loss, prog_bar=True)
        self.log("training/state_mae_rad", self._last_state_mae_rad)
        self.log("training/mem_valid_frac", self._last_mem_valid_frac)
        for k, v in (self._last_flow_by_t or {}).items():
            self.log(f"training/flow_t_{k}", v)
        if self._last_scwl is not None:
            for k, v in self._last_scwl.items():
                self.log(f"training/scwl_{k}", v)
        loss = flow + self.state_pred_weight * self._last_state_loss
        if self._last_mcp_losses:
            for i, ml in enumerate(self._last_mcp_losses):
                self.log(f"training/mcp_loss_k{i + 1}", ml, prog_bar=(i == 0))
                wk = self.mcp_loss_weights[i] if i < len(self.mcp_loss_weights) else 0.0
                loss = loss + float(wk) * ml
        if self._last_contact_bce is not None:
            self.log("training/contact_bce", self._last_contact_bce, prog_bar=True)
            loss = loss + self.contact_pred_weight * self._last_contact_bce
            self.log("training/contact_acc", self._last_contact_acc)
            self.log("training/contact_acc_left", self._last_contact_acc_l)
            self.log("training/contact_acc_right", self._last_contact_acc_r)
            self.log("training/contact_acc_chance", self._last_contact_chance)
            if self._last_contact_persist is not None:
                self.log("training/contact_acc_persist", self._last_contact_persist)
        self.log("training/loss", loss, prog_bar=True)
        return {"loss": loss}

    # ===================== EMA ============================================ #
    def _ema_named_params(self):
        return [(n.replace("_orig_mod.", ""), p)
                for n, p in self.dynamics.named_parameters() if p.requires_grad]

    @torch.no_grad()
    def _ema_update(self) -> None:
        """ema = d*ema + (1-d)*live with d = min(ema_decay, (1+step)/(10+step)) under ema_warmup."""
        if self.ema_decay <= 0.0:
            return
        d = self.ema_decay
        if self.ema_warmup:
            step = int(self.global_step)
            d = min(d, (1.0 + step) / (10.0 + step))
        self._last_ema_decay = d
        if self._ema_state is None:
            self._ema_state = {n: p.detach().clone().float() for n, p in self._ema_named_params()}
            return
        if not getattr(self, "_ema_dev_aligned", False):
            for n, p in self._ema_named_params():
                self._ema_state[n] = self._ema_state[n].to(p.device)
            self._ema_dev_aligned = True
        for n, p in self._ema_named_params():
            self._ema_state[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def _ema_swap_in(self) -> bool:
        if self.ema_decay <= 0.0 or self._ema_state is None:
            return False
        self._ema_backup = {n: p.detach().clone() for n, p in self._ema_named_params()}
        for n, p in self._ema_named_params():
            p.copy_(self._ema_state[n].to(p.dtype))
        return True

    @torch.no_grad()
    def _ema_swap_out(self) -> None:
        if self._ema_backup is None:
            return
        for n, p in self._ema_named_params():
            p.copy_(self._ema_backup[n])
        self._ema_backup = None

    def on_train_batch_end(self, *args, **kwargs) -> None:
        # One EMA update per OPTIMIZER step (not per micro-batch under gradient accumulation).
        try:
            stepped = not self.trainer.fit_loop.epoch_loop._should_accumulate()
        except Exception:
            stepped = True
        if not stepped:
            return
        self._ema_update()
        if self.ema_decay > 0.0 and self.global_step % 100 == 0:
            self.log("training/ema_decay_eff", float(self._last_ema_decay))

    def on_validation_start(self) -> None:
        self._ema_active = self._ema_swap_in()
        self._ar_ids = None

    def on_validation_end(self) -> None:
        if getattr(self, "_ema_active", False):
            self._ema_swap_out()
            self._ema_active = False

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if getattr(self, "_ema_active", False):            # never save EMA weights as the live ones
            self._ema_swap_out()
            self._ema_active = False
        if self._ema_state is not None:
            checkpoint["ema"] = {n: v.detach().cpu() for n, v in self._ema_state.items()}

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        ema = checkpoint.get("ema", None)
        if ema is not None and self.ema_decay > 0.0:
            self._ema_state = {n: v.clone().float() for n, v in ema.items()}

    # ===================== validation ===================================== #
    def _ar_batch_ids(self) -> set:
        """Seeded random sample of val batch indices that get the AR rollout metric."""
        k = self.val_stream_ar_batches
        if k <= 0:
            return set()
        if self._ar_ids is not None:
            return self._ar_ids
        n = 0
        nb = getattr(getattr(self, "trainer", None), "num_val_batches", None)
        if isinstance(nb, (list, tuple)) and nb:
            nb = nb[0]
        if isinstance(nb, (int, float)) and 0 < float(nb) < float("inf"):
            n = int(nb)
        if n <= 0:
            self._ar_ids = set(range(k))
        elif n <= k:
            self._ar_ids = set(range(n))
        else:
            self._ar_ids = set(random.Random(self.val_sample_seed).sample(range(n), k))
        return self._ar_ids

    @torch.no_grad()
    def _log_val_flow_buckets_stream(self, batch: dict, namespace: str, batch_idx: int) -> None:
        """Teacher-forced training objective at fixed flow-times with reproducible noise."""
        z = self.stack_views(batch["latent"])
        T, B = z.shape[:2]
        if T < 2:
            return
        action = rearrange(self.cond_action(batch["action"]), "b t a -> t b a")
        state = rearrange(batch["state"].float(), "b t a -> t b a")
        if "latent_memory" in batch and "mem_dt" in batch:
            z_mem = self.stack_views(batch["latent_memory"])
            mem_state_norm = self.normalizer["state"].normalize(
                rearrange(batch["state_memory"].float(), "b m a -> m b a"))
            mem_dt = rearrange(batch["mem_dt"].float(), "b m -> m b")
            mem_valid = rearrange(batch["mem_valid"], "b m -> m b")
        else:
            z_mem = z.new_zeros((0, *z.shape[1:]))
            mem_state_norm = state.new_zeros((0, B, self.pose_dim))
            mem_dt = z.new_zeros((0, B))
            mem_valid = torch.ones((0, B), dtype=torch.bool, device=z.device)
        fidx = rearrange(batch["frame_idx"].float(), "b t -> t b")
        base_seed = int(fidx[0, 0].item()) if fidx.numel() else batch_idx
        for bi, (lo, hi) in enumerate(self.VAL_FLOW_BUCKETS):
            g = torch.Generator(device=z.device).manual_seed(1_000_003 * base_seed + 7919 * bi)
            flow = self._flow_loss_stream(z, action, state, z_mem, mem_state_norm, mem_dt,
                                         mem_valid, fixed_t=0.5 * (lo + hi), gen=g)
            self.log(f"{namespace}/flow_t{lo:g}_{hi:g}", flow, sync_dist=True)
            if bi == len(self.VAL_FLOW_BUCKETS) // 2:
                self.log(f"{namespace}/state_loss", self._last_state_loss, sync_dist=True)
                self.log(f"{namespace}/state_mae_rad", self._last_state_mae_rad, sync_dist=True)
                self.log(f"{namespace}/dyn_latent_mse_tf", self._last_x0_mse, sync_dist=True, prog_bar=True)
                self.log(f"{namespace}/dyn_cos_sim_tf", self._last_x0_cos, sync_dist=True, prog_bar=True)

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        namespace = "validation"
        self._log_val_flow_buckets_stream(batch, namespace, batch_idx)
        if batch_idx not in self._ar_batch_ids():
            return
        from dreamfar.inference.seeding import build_regime
        z_bt = rearrange(self.stack_views(batch["latent"]), "t b c h w -> b t c h w")
        T = z_bt.shape[1]
        act = self.cond_action(batch["action"])
        st = batch["state"].to(self.device)
        fidx = batch["frame_idx"].to(self.device)
        k = build_regime("dup", self, z_bt, act, st, fidx)          # cold start, duplicated frame 0
        pad, seed_latents = int(k["pad"]), k["seed_latents"]
        act, st, fidx = k["action"], k["state"], k["frame_idx"]
        n_ar = max(1, self.val_ar_rows)
        if z_bt.shape[0] > n_ar:
            z_bt, act, st, fidx, seed_latents = (z_bt[:n_ar], act[:n_ar], st[:n_ar], fidx[:n_ar],
                                                 seed_latents[:n_ar])
        z_pred, _ = self.rollout(z_bt[:, :1], act, st, fidx, seed_latents=seed_latents, compiled=True)
        z_pred = z_pred[:, pad:pad + T - 1]
        z_gt = z_bt[:, 1:T]
        self.log(f"{namespace}/dyn_latent_mse", F.mse_loss(z_pred, z_gt), sync_dist=True, prog_bar=True)
        self.log(f"{namespace}/dyn_cos_sim", F.cosine_similarity(z_pred, z_gt, dim=2).mean(),
                 sync_dist=True, prog_bar=True)

    # ===================== inference: memory retrieval ==================== #
    def _mem_key_query(self, states: torch.Tensor, actions: torch.Tensor, pool: np.ndarray, q: int,
                       key_cache: Optional[dict] = None) -> tuple:
        """(keys (m,D), query (1,D)) in EE space: keys = FK of the committed frames' predicted joints
        (cached per frame), query = the commanded EE action of the frame being decoded."""
        qa = np.atleast_1d(np.asarray(q, dtype=np.int64))
        qi = torch.as_tensor(np.clip(qa - 1, 0, actions.shape[0] - 1), device=actions.device)
        qv = actions[qi].detach().float().cpu().numpy()

        def fn(p):
            return self._ee_fk(states[p].detach().float().cpu().numpy().astype(np.float64))
        if key_cache is None:
            return fn(pool), qv
        new = [int(f) for f in pool if int(f) not in key_cache]
        if new:
            for f, r in zip(new, fn(np.asarray(new, dtype=np.int64))):
                key_cache[f] = r
        return np.stack([key_cache[int(f)] for f in pool]), qv

    def _select_memory_stream_infer(self, kf: np.ndarray, states: torch.Tensor, c0: int, q: int,
                                    actions: torch.Tensor, key_cache: Optional[dict] = None) -> tuple:
        """Inference twin of the loader's streaming memory selection: slot 0 = frame 0 (anchor), the
        rest = top-(M-1) keyframes <= c0 by the shared retrieval score, MASK-padded.
        Returns (idx[M], valid[M], dt[M]) with dt in model frames from c0."""
        M = self.n_memory
        fs = float(self.frame_skip)
        bank = kf[kf <= c0] if c0 >= 0 else kf[:0]
        sel = [0]
        pool = bank[bank != 0]
        if pool.size and M > 1:
            cand, queries = self._mem_key_query(states, actions, pool, q, key_cache=key_cache)
            conf = keyframe_confidence(
                cand, queries, pool, max(c0, 1),
                sim_weight=self.kf_sim_weight, time_weight=self.kf_time_weight,
                normalize_sim=self.kf_normalize_sim, dt_ref=self.kf_dt_ref)
            k = min(M - 1, pool.size)
            top = np.argpartition(-conf, k - 1)[:k] if pool.size > k else np.arange(pool.size)
            sel += [int(f) for f in np.sort(pool[top])]
        n = len(sel)
        valid = [True] * n + [False] * (M - n)
        sel = sel + [0] * (M - n)
        dt = [max(0.0, (c0 - sel[i]) / fs) for i in range(M)]
        return sel, valid, dt

    # ===================== inference: KV-cached streaming rollout ========= #
    @torch.no_grad()
    def rollout(self, z_hist: torch.Tensor, action: torch.Tensor, states: torch.Tensor,
                frame_idx: torch.Tensor, seed_latents: Optional[torch.Tensor] = None,
                seed_contact: Optional[torch.Tensor] = None, compiled: bool = False,
                on_commit: Optional[Callable[[int, torch.Tensor], None]] = None,
                pending_fn: Optional[Callable[[], int]] = None,
                on_admit: Optional[Callable[[int], None]] = None,
                wait_fn: Optional[Callable[[], bool]] = None):
        """KV-cached streaming rollout. z_hist:(B,1,C,V*h,w) GT frame 0; action:(B,total,A) normalized
        conditioning; states:(B,total,pose) raw (seed rows used); frame_idx:(B,total) (length only).
        seed_latents:(B,n_seed,...) GT prefix; seed_contact:(n_seed,2) its per-arm contact.

        Default mode keeps P frames in flight and commits one per step. Interactive mode (pending_fn
        given): slots are admitted only when an action is pending (`on_admit(frame)` lets the caller
        write that frame's action row), the pyramid drains when input stops and `wait_fn()` blocks
        while idle (return False to end). `on_commit(frame, latent)` fires at every commit.
        Returns (predicted latents (B,total-1,C,V*h,w), per-commit memory selections)."""
        z_hist = rearrange(z_hist, "b t c h w -> t b c h w")
        act = rearrange(action, "b t a -> t b a")
        st_tb = rearrange(states, "b t a -> t b a")
        total = act.shape[0]
        B = z_hist.shape[1]
        T, P, M = self.n_tokens, self.dyn_flow_steps, self.n_memory
        L = P
        nsk = 1
        n_cmt_max = T - L
        assert n_cmt_max >= 1, f"n_tokens={T} must exceed the pyramid depth {P}"
        s_lvl = self.stabilization_level
        ts = self.flow_timescale
        nrm = self.normalizer["state"]
        dev = z_hist.device
        wdt = z_hist.dtype                                    # latent / scratch dtype
        dit = self._cm
        pdt = next(dit.parameters()).dtype                    # DiT weight dtype (inputs cast at the call)
        ad = self._action_dim
        # compiled: pad the committed cache to n_cmt_max so the per-step decode has a static shape.
        pad_to = n_cmt_max if compiled else None
        if compiled and getattr(self, "_cmp_stream_decode", None) is None:
            self._cmp_stream_decode = torch.compile(dit.stream_decode)
        decode_fn = self._cmp_stream_decode if compiled else dit.stream_decode
        if compiled and getattr(self, "_cmp_stream_snap", None) is None:
            self._cmp_stream_snap = torch.compile(dit.stream_decode)      # q=1 snapshot shape
        snap_fn = self._cmp_stream_snap if compiled else None

        xs = z_hist.new_zeros((total, *z_hist.shape[1:]))
        st_pred = st_tb.clone()
        n_seed = 1 if seed_latents is None else int(seed_latents.shape[1])
        xs[:n_seed] = (z_hist[:n_seed] if seed_latents is None
                       else rearrange(seed_latents, "b t c h w -> t b c h w"))
        g = torch.Generator(device=dev).manual_seed(1234)
        cn = torch.clamp(torch.randn(xs.shape, generator=g, device=dev, dtype=xs.dtype),
                         -self.clip_noise, self.clip_noise)
        sn = torch.clamp(torch.randn(st_pred.shape, generator=g, device=dev, dtype=xs.dtype),
                         -self.clip_noise, self.clip_noise)
        xf = torch.clamp(torch.randn((L, *z_hist.shape[1:]), generator=g, device=dev,
                                     dtype=xs.dtype), -self.clip_noise, self.clip_noise)
        sf = torch.clamp(torch.randn((L, B, self.pose_dim), generator=g, device=dev,
                                     dtype=xs.dtype), -self.clip_noise, self.clip_noise)
        levels, dts = self._infer_t_levels(P)
        t_born = levels[0]
        tf = torch.full((L,), t_born, device=dev)
        nf = [0] * L                                          # fine steps taken per slot
        af = list(range(n_seed, n_seed + L))                  # frame index per slot
        occ_mode = pending_fn is not None
        n_occ = 0 if occ_mode else L
        next_f = n_seed if occ_mode else n_seed + L
        if occ_mode:
            af = [n_seed] * L
        # Flow-time each committed frame's KV is taken at: scalar, or per-frame draw from a list
        # with training-frequency weights (seeded so reruns match).
        _cn = self.stream_ctx_noise
        if isinstance(_cn, list):
            _w = self._ctx_noise_weights(_cn)
            _lv = sorted(float(x) for x in _cn)
            ctx_lv = np.random.default_rng(1234).choice(_lv, size=total + P + 8, p=_w)
            ctx_t = float(np.mean(ctx_lv))
        else:
            ctx_t = float(_cn) if _cn is not None else s_lvl
            ctx_lv = np.full(total + P + 8, ctx_t, dtype=float)
        mem_lvl = s_lvl

        def _lv_of(i: int) -> float:
            return float(ctx_lv[min(max(i, 0), len(ctx_lv) - 1)])

        # Harvest: a committed frame's KV at a level that is one of the P query levels is taken from
        # the in-flight pass instead of a dedicated snapshot forward.
        _pred_lv = set(ctx_lv[n_seed:].tolist()) or set(ctx_lv.tolist())
        _grid_lv = {x for x in _pred_lv if x > 0 and any(abs(x - lv) < 1e-6 for lv in levels[:P])}
        snap_reuse = bool(self.stream_snap_reuse and _grid_lv)
        snap_by_frame: dict = {}
        n_snap_miss = 0

        c_prob, c_lr = {}, {}
        if seed_contact is not None:
            for f in range(min(n_seed, seed_contact.shape[0])):
                l, r = float(seed_contact[f, 0]), float(seed_contact[f, 1])
                c_lr[f] = (l, r); c_prob[f] = max(l, r)
        mem_sel_log = []
        ro_slot = self._contact_readout_slot(levels)
        kf_track = ContactKeyframeTracker(n_arms=2, dwell=self.contact_min_dwell)
        mem_keys: dict = {}

        sink_cache = None
        if nsk:
            z0 = z_hist.new_zeros((B, z_hist.shape[2], 1, *z_hist.shape[3:]))
            s0 = st_tb.new_zeros((1, B, self.pose_dim))
            sink_cache = dit.stream_prefill(
                z0.to(pdt), torch.zeros((1, B), device=dev, dtype=wdt),
                torch.zeros((1, B, ad), device=dev, dtype=pdt), s0.to(pdt),
                rope_pos=torch.zeros(1, device=dev),
                attn_mask=torch.ones((B, 1, 1), dtype=torch.bool, device=dev),
                cond_mask=torch.zeros((B, 1), device=dev, dtype=pdt), n_sink=1, n_mem=0)
        mem_pool: OrderedDict = OrderedDict()                # keyframe -> per-block KV, LRU order
        pool_cap = max(self.stream_kv_pool_resident, M, 1)
        self._stream_dbg = {"st_pred": st_pred, "mem_sel_log": mem_sel_log, "mem_pool": mem_pool,
                            "ctx_lv": ctx_lv}

        def _mem_kv(f: int):
            kv = mem_pool.get(f)
            if kv is None:
                mx = ((1 - s_lvl) * xs[f] + s_lvl * cn[f]).unsqueeze(2)          # (B,C,1,h,w)
                msn = ((1 - s_lvl) * nrm.normalize(st_pred[f]) + s_lvl * sn[f]).unsqueeze(0)
                kv = mem_pool[f] = dit.stream_prefill(
                    mx.to(pdt), torch.full((1, B), mem_lvl * ts, device=dev, dtype=wdt),
                    torch.zeros((1, B, ad), device=dev, dtype=pdt), msn.to(pdt),
                    rope_pos=torch.zeros(1, device=dev),
                    attn_mask=torch.ones((B, 1, 1), dtype=torch.bool, device=dev),
                    cond_mask=torch.zeros((B, 1), device=dev, dtype=pdt), n_sink=0, n_mem=1)
                while len(mem_pool) > pool_cap:
                    mem_pool.popitem(last=False)
            else:
                mem_pool.move_to_end(f)
            return kv

        cmt = list(range(n_seed))
        cmt_cache: list = []
        # Seed prefill: only the last n_cmt_max seeded frames are ever read; snapshot each against
        # [sink | memory | previous committed] with a padded (static) context width.
        seed_lo = max(0, n_seed - n_cmt_max)
        kf_seed = kf_track.advance(c_lr, n_seed)
        for fi in cmt:
            if fi < seed_lo:
                continue
            if M > 0:
                m_idx, m_ok, m_dt = self._select_memory_stream_infer(
                    kf_seed, st_pred[:, 0], fi, fi, actions=act[:, 0], key_cache=mem_keys)
                mem_kvs_pf = [_mem_kv(int(f)) for f in m_idx]
                mem_pos_pf = dt_to_rope_pos(torch.tensor(m_dt, device=dev, dtype=torch.float32),
                                            dt_span=self.stream_dt_span, dt_ref=self.stream_dt_ref)
            else:
                mem_kvs_pf = []
                mem_pos_pf = torch.zeros(0, device=dev)
            win_pf = cmt_cache[-n_cmt_max:] if len(cmt_cache) > n_cmt_max else cmt_cache
            n_c_pf = len(win_pf)
            ctx_k, ctx_v, ctx_pos, n_ctx = _assemble_ctx(
                sink_cache, mem_kvs_pf, win_pf, nsk, M, n_c_pf, dev, mem_pos_pf, pad_to)
            cmt_cache.append(_snapshot_committed(
                dit, xs[fi], st_pred[fi], act, fi, total, _lv_of(fi), ts, ad, nrm, sn, cn,
                ctx_k, ctx_v, ctx_pos, n_ctx, n_c_pf, dev, B,
                n_real=nsk + M + n_c_pf, decode_fn=snap_fn))
        kf_bank = kf_seed

        step = 0
        last_commit_step = 0
        while (next_f < total or n_occ > 0) if occ_mode else (af[0] < total):
            if occ_mode:
                while n_occ == 0 and pending_fn() <= 0:          # idle: nothing in flight or pending
                    if wait_fn is None or not wait_fn():
                        return (rearrange(xs[1:total], "t b c h w -> b t c h w"), mem_sel_log)
                while n_occ < L and next_f < total and pending_fn() > 0:   # admit pending actions
                    xf[n_occ] = torch.clamp(torch.randn(xf.shape[1:], generator=g, device=dev,
                                                        dtype=xf.dtype), -self.clip_noise, self.clip_noise)
                    sf[n_occ] = torch.clamp(torch.randn((B, self.pose_dim), generator=g, device=dev,
                                                        dtype=sf.dtype), -self.clip_noise, self.clip_noise)
                    tf[n_occ] = t_born
                    nf[n_occ] = 0
                    af[n_occ] = next_f
                    if on_admit is not None:
                        on_admit(next_f)
                    next_f += 1
                    n_occ += 1
                if n_occ == 0:
                    continue
            win = cmt[-n_cmt_max:] if len(cmt) > n_cmt_max else cmt
            n_c = len(win)
            win_cache = cmt_cache[-n_cmt_max:] if len(cmt_cache) > n_cmt_max else cmt_cache
            assert n_c + L <= T
            c0 = win[0]
            if M > 0:
                q = min(af[0], total - 1)
                m_idx, m_ok, m_dt = self._select_memory_stream_infer(kf_bank, st_pred[:, 0], c0, q,
                                                                     actions=act[:, 0], key_cache=mem_keys)
                cur_mem = [f for f, ok in zip(m_idx, m_ok) if ok]
                mem_kvs = [_mem_kv(int(f)) for f in m_idx]
                mem_pos = dt_to_rope_pos(torch.tensor(m_dt, device=dev, dtype=torch.float32),
                                         dt_span=self.stream_dt_span, dt_ref=self.stream_dt_ref)
                mem_valid = torch.tensor(m_ok, device=dev)
            else:
                cur_mem = []; mem_kvs = []; mem_pos = torch.zeros(0, device=dev)
                mem_valid = torch.ones(0, dtype=torch.bool, device=dev)
            ctx_k, ctx_v, ctx_pos, n_ctx = _assemble_ctx(
                sink_cache, mem_kvs, win_cache, nsk, M, n_c, dev, mem_pos, pad_to=pad_to)
            # decode mask: L in-flight rows over n_ctx+L columns (sink, valid memory, real committed,
            # causal in-flight; padded committed columns and empty slots off)
            dmask = torch.zeros((B, L, n_ctx + L), dtype=torch.bool, device=dev)
            dmask[:, :, :nsk] = True
            if M:
                dmask[:, :, nsk:nsk + M] = mem_valid[None, None, :].expand(B, L, M)
            dmask[:, :, nsk + M:nsk + M + n_c] = True
            _ifm = torch.ones((L, L), device=dev).tril().bool()
            if occ_mode and n_occ < L:
                _ifm = _ifm.clone(); _ifm[:, n_occ:] = False
            dmask[:, :, n_ctx:] = _ifm[None]
            q_pos = torch.arange(n_c, n_c + L, device=dev, dtype=torch.float32)
            a_src = [min(total - 1, max(0, i - 1)) for i in af]
            cond_if = act[a_src].clone()
            for p, i in enumerate(af):
                if i == 0:
                    cond_if[p] = 0.0
            x_if = rearrange(xf, "p b c h w -> b c p h w").to(pdt)
            v_img, v_st, c_out, if_kv = decode_fn(
                x_if, (tf.unsqueeze(1).expand(L, B) * ts).to(wdt), cond_if.to(pdt), sf.to(pdt),
                [(ck, cv) for ck, cv in zip(ctx_k, ctx_v)], ctx_pos, q_pos, dmask,
                cond_mask=torch.ones((B, L), device=dev, dtype=pdt))
            v_img = rearrange(v_img, "b c p h w -> p b c h w")
            # contact readout at the slot nearest contact_infer_t
            j = min(ro_slot, L - 1)
            pr = torch.sigmoid(c_out[j, 0].float())
            fidx_ro = af[j]
            if fidx_ro < total:
                c_lr[fidx_ro] = (float(pr[0]), float(pr[1])); c_prob[fidx_ro] = float(pr.max())
            # harvest committed-KV snapshots from slots sitting at their frame's ctx level (before the
            # advance); columns per frame differ per block type, so slice per block
            if snap_reuse:
                for j in range(L):
                    if abs(float(tf[j]) - _lv_of(af[j])) < 1e-6 and af[j] not in snap_by_frame:
                        snap_by_frame[af[j]] = [
                            (k[:, :, j * (k.shape[2] // L):(j + 1) * (k.shape[2] // L)].clone(),
                             v[:, :, j * (v.shape[2] // L):(j + 1) * (v.shape[2] // L)].clone())
                            for k, v in if_kv]
            # advance: the staircase ramp (default) or every occupied slot (interactive)
            n_adv = n_occ if occ_mode else min(step + 1, L)
            for j in range(n_adv):
                dt_step = min(dts[min(nf[j], P - 1)], float(tf[j]))
                xf[j] = xf[j] + dt_step * v_img[j]
                sf[j] = sf[j] + dt_step * v_st[j]
                tf[j] = tf[j] - dt_step
                nf[j] += 1
            # commit every clean slot
            while (n_occ > 0 if occ_mode else True) and float(tf[0]) <= 1e-6:
                i = af[0]
                if i < total:
                    xs[i] = self._renorm_per_view(xf[0:1])[0]       # unit-sphere re-projection
                    st_pred[i] = nrm.unnormalize(sf[0])
                    if on_commit is not None:
                        on_commit(i, xs[i])
                    sk = snap_by_frame.pop(i, None) if snap_reuse else None
                    if sk is None:
                        if snap_reuse:
                            n_snap_miss += 1
                        sk = _snapshot_committed(
                            dit, xs[i], st_pred[i], act, i, total, _lv_of(i), ts, ad, nrm, sn, cn,
                            ctx_k, ctx_v, ctx_pos, n_ctx, n_c, dev, B,
                            n_real=nsk + M + n_c, decode_fn=snap_fn)
                    cmt.append(i); cmt_cache.append(sk); mem_sel_log.append(cur_mem)
                    last_commit_step = step
                    del cmt_cache[:-n_cmt_max]                       # only the window is ever read
                    kf_bank = kf_track.advance(c_lr, i + 1)
                xf = torch.cat([xf[1:], torch.clamp(torch.randn(
                    (1, *xf.shape[1:]), generator=g, device=dev, dtype=xf.dtype),
                    -self.clip_noise, self.clip_noise)], 0)
                sf = torch.cat([sf[1:], torch.clamp(torch.randn(
                    (1, B, self.pose_dim), generator=g, device=dev, dtype=sf.dtype),
                    -self.clip_noise, self.clip_noise)], 0)
                tf = torch.cat([tf[1:], torch.full((1,), t_born, device=dev)], 0)
                nf = nf[1:] + [0]
                af = af[1:] + [af[-1] + 1]
                if occ_mode:
                    n_occ -= 1
            step += 1
            if occ_mode:
                if n_occ > 0 and step - last_commit_step > 4 * P + 8:
                    raise RuntimeError(f"streaming rollout stalled (step {step}, af0={af[0]}, n_occ={n_occ})")
            elif step > total + L + 8:
                raise RuntimeError(f"streaming rollout failed to advance (step {step}, af0={af[0]})")
        z_pred = rearrange(xs[1:total], "t b c h w -> b t c h w")
        self._last_contact_lr = dict(c_lr)
        self._last_kf_frames = kf_track.advance(c_lr, total).tolist()
        self._last_mem_sel = mem_sel_log
        self._last_ctx_t = ctx_t
        self._last_snap_miss = n_snap_miss
        self._last_state_pred = st_pred
        return z_pred, mem_sel_log

    # ===================== optimizer ====================================== #
    def configure_optimizers(self):
        cfg = self.cfg
        lr = float(cfg.lr)
        wd = float(cfg.weight_decay)
        betas = tuple(cfg.optimizer_beta)
        eps = float(cfg.optimizer_eps)
        # AdamW with decoupled weight decay on the hidden weight matrices ONLY (attention qkv/out +
        # SwiGLU); norms, biases, gates, embedders and heads get wd=0.
        mats, others = split_muon_adamw_params(self.dynamics.named_parameters())
        n_dyn = sum(1 for p in self.dynamics.parameters() if p.requires_grad)
        assert len(mats) + len(others) == n_dyn, \
            f"wd split leak: mats({len(mats)}) + others({len(others)}) != dynamics({n_dyn})"
        groups = [{"params": mats, "weight_decay": wd}, {"params": others, "weight_decay": 0.0}]
        print(f"[optimizer] AdamW wd={wd} on {len(mats)} matrices "
              f"({sum(p.numel() for p in mats) / 1e6:.1f}M); wd=0 on {len(others)} params "
              f"({sum(p.numel() for p in others) / 1e6:.1f}M) | lr={lr:.2e} betas={betas}", flush=True)
        opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
        warmup = int(cfg.warmup_steps)
        lr_min, decay_start = cfg.lr_min, cfg.lr_decay_start
        if lr_min is not None and decay_start is not None:
            # Anchored schedule over the absolute optimizer step: linear warmup, hold `lr` until
            # lr_decay_start, cosine to lr_min over lr_decay_steps, then hold lr_min.
            from torch.optim.lr_scheduler import LambdaLR
            start = int(decay_start)
            n = int(cfg.lr_decay_steps or 0) or max(1, int(self.trainer.max_steps) - start)
            floor = float(lr_min) / lr

            def _anchored_cosine(step: int) -> float:
                if warmup > 0 and step < warmup:
                    return 1e-3 + (1.0 - 1e-3) * step / warmup
                if step < start:
                    return 1.0
                if step >= start + n:
                    return floor
                return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * (step - start) / n))
            sched = LambdaLR(opt, lr_lambda=_anchored_cosine)
            print(f"[lr_sched] warmup={warmup}, hold lr={lr:.2e} until step {start}, cosine over {n} "
                  f"steps to {float(lr_min):.2e}, then hold", flush=True)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        if warmup > 0:
            from torch.optim.lr_scheduler import LinearLR
            sched = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return opt

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log the pre-clip gradient norm; abort on a non-finite one (never write a NaN checkpoint)."""
        norms = [p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None]
        if not norms:
            return
        total = torch.linalg.vector_norm(torch.stack([n.float() for n in norms]))
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"non-finite gradient norm ({total.item()}) at global_step "
                               f"{self.global_step}: the run has diverged")
        if self.logger is not None:
            self.log("training/grad_norm", total, prog_bar=False, rank_zero_only=True)

    def on_train_start(self) -> None:
        self._t_start = time.time()
        self._samples_seen = 0

    def _log_progress(self, batch_size: int, batch_idx: int) -> None:
        if self._t_start is None:
            self._t_start = time.time()
        self._samples_seen += batch_size
        if batch_idx % 50 == 0:
            el = max(time.time() - self._t_start, 1e-9)
            self.log("training/samples_per_s", self._samples_seen / el)
