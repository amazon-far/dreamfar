"""DiT latent dynamics: a stack of transformer blocks over the (B,C,T,H,W) latent grid plus one state
token per frame, each block = attention (factorized spatial+temporal, or joint spatio-temporal) +
AdaLN-Zero SwiGLU MLP. Streaming layout ``[sink | memory | context]`` with a KV-cached decode path.
"""
from __future__ import annotations

from functools import partial
from typing import List, Optional, Sequence, Tuple

import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import (
    FullAttentionBlockWithState,
    SpatialAttentionBlockWithState,
    TemporalAttentionBlockWithState,
)
from .embeddings import TimestepEmbedding, Timesteps
from .mcp import McpHeads, mcp_tap_layers_for_depth


class _SwiGLU(nn.Module):
    """SwiGLU FFN on channels-last tokens: ``c_proj(a * silu(b))`` with ``a, b = c_fc(x).chunk(2)``.
    hidden = int(dim*mlp_ratio) padded UP to a multiple of ``hidden_multiple`` with exactly-inert
    zero units (zero c_fc rows/bias in both halves, zero c_proj columns)."""

    def __init__(self, dim: int, mlp_ratio: float = 8 / 3, hidden_multiple: int = 8):
        super().__init__()
        self.hidden = int(dim * mlp_ratio)                       # the logical (unpadded) width
        m = max(1, int(hidden_multiple))
        self.hidden_pad = -(-self.hidden // m) * m               # ceil to a multiple of m
        self.c_fc = nn.Linear(dim, self.hidden_pad * 2)
        self.c_proj = nn.Linear(self.hidden_pad, dim)
        if self.hidden_pad != self.hidden:
            with torch.no_grad():                                # pads start (and stay) exactly zero
                h, hp = self.hidden, self.hidden_pad
                self.c_fc.weight[h:hp].zero_(); self.c_fc.weight[hp + h:].zero_()
                self.c_fc.bias[h:hp].zero_(); self.c_fc.bias[hp + h:].zero_()
                self.c_proj.weight[:, h:].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.c_fc(x).chunk(2, dim=-1)
        return self.c_proj(a * nn.functional.silu(b))


class _DiTBlockWithState(nn.Module):
    """Factorized block: spatial-attn -> temporal-attn -> AdaLN-Zero SwiGLU MLP, all gated residuals,
    carrying the image grid x (B,C,T,H,W) and the side stream s (B,C,S,T) (S=1 state token) jointly.
    The MLP + FiLM are the SAME modules applied to both streams."""

    def __init__(self, dim: int, emb_dim: int, cond_dim: int, spatial_attn_ws, temporal_attn_ws,
                 mlp_ratio: float = 8 / 3, mlp_hidden_multiple: int = 8):
        super().__init__()
        self.spatial = spatial_attn_ws(dim)
        self.temporal = temporal_attn_ws(dim)
        # per-TOKEN channel RMSNorm (over C only) — causal-safe, no cross-frame stats
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = _SwiGLU(dim, mlp_ratio=mlp_ratio, hidden_multiple=mlp_hidden_multiple)
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim * 2))
        self.cond_emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))
        # AdaLN-Zero: zero-init per-channel gates + zero-init modulation -> identity block at init
        self.gate_spatial = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.gate_temporal = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.gate_mlp = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        nn.init.zeros_(self.emb_layers[-1].weight); nn.init.zeros_(self.emb_layers[-1].bias)
        nn.init.zeros_(self.cond_emb_layers[-1].weight); nn.init.zeros_(self.cond_emb_layers[-1].bias)

    def _adaln_mlp(self, x, emb, cond, cond_mask=None, tok_emb_mask=None):
        """AdaLN-modulated SwiGLU on a (B,C,T,H,W) tensor (the grid, or the side stream viewed as
        (B,C,T,S,1)). cond_mask (B,T) in {0,1} zeroes the action FiLM per frame; tok_emb_mask
        (channel-first 5-D, {0,1}) zeroes the noise-level FiLM per token (exact identity)."""
        # channels-last (B,T,H,W,C) from the norm through the FiLM and the Linear SwiGLU
        h = self.norm(rearrange(x, "b c t h w -> b t h w c"))
        scale, shift = rearrange(self.emb_layers(emb), "b t c -> b t 1 1 c").chunk(2, dim=-1)
        if tok_emb_mask is not None:
            tm = tok_emb_mask.permute(0, 2, 3, 4, 1)
            scale = scale * tm
            shift = shift * tm
        h = h * (1 + scale) + shift
        cscale, cshift = rearrange(self.cond_emb_layers(cond).type(h.dtype), "b t c -> b t 1 1 c").chunk(2, dim=-1)
        if cond_mask is not None:
            m = cond_mask.type(h.dtype).view(cond_mask.shape[0], cond_mask.shape[1], 1, 1, 1)  # (B,T,1,1,1)
            cscale = cscale * m
            cshift = cshift * m
        h = h * (1 + cscale) + cshift
        return x + self.gate_mlp * rearrange(self.mlp(h), "b t h w c -> b c t h w")

    def forward(self, x, s, emb, cond, attn_mask=None, rope_pos=None, cond_mask=None,
                frame_emb_mask=None):
        # frame_emb_mask (1,1,T,1,1) in {0,1}: zeroes the noise-level FiLM on frames that carry no
        # noised content (the sink); T sits on axis 2 of both the grid and the (B,C,T,S,1) side view.
        xa, sa = self.spatial(x, s)
        x = x + self.gate_spatial * (xa - x)
        s = s + self.gate_spatial.view(1, -1, 1, 1) * (sa - s)
        xt, st = self.temporal(x, s, attn_mask=attn_mask, rope_pos=rope_pos)
        x = x + self.gate_temporal * (xt - x)
        s = s + self.gate_temporal.view(1, -1, 1, 1) * (st - s)
        x = self._adaln_mlp(x, emb, cond, cond_mask=cond_mask, tok_emb_mask=frame_emb_mask)
        s_in = rearrange(s, "b c s t -> b c t s 1")
        s = rearrange(self._adaln_mlp(s_in, emb, cond, cond_mask=cond_mask,
                                     tok_emb_mask=frame_emb_mask), "b c t s 1 -> b c s t")
        return x, s

    # ---- streaming KV cache: prefill (full forward + capture temporal K/V) and decode (in-flight) --
    def forward_prefill(self, x, s, emb, cond, attn_mask=None, rope_pos=None, cond_mask=None,
                        frame_emb_mask=None):
        """Run the block over CONTEXT frames AND capture its UN-ROPED temporal K/V (post-spatial, where
        the temporal attention reads them). Returns (x, s, k_unroped, v), k/v packed (rows,heads,T,d)."""
        xa, sa = self.spatial(x, s)
        x = x + self.gate_spatial * (xa - x)
        s = s + self.gate_spatial.view(1, -1, 1, 1) * (sa - s)
        k_u, v = self.temporal.stream_encode(x, s)             # un-roped, post-spatial (cache entry)
        xt, st = self.temporal(x, s, attn_mask=attn_mask, rope_pos=rope_pos)
        x = x + self.gate_temporal * (xt - x)
        s = s + self.gate_temporal.view(1, -1, 1, 1) * (st - s)
        x = self._adaln_mlp(x, emb, cond, cond_mask=cond_mask, tok_emb_mask=frame_emb_mask)
        s_in = rearrange(s, "b c s t -> b c t s 1")
        s = rearrange(self._adaln_mlp(s_in, emb, cond, cond_mask=cond_mask, tok_emb_mask=frame_emb_mask),
                      "b c t s 1 -> b c s t")
        return x, s, k_u, v

    def forward_stream(self, x, s, emb, cond, ctx_k, ctx_v, ctx_pos, q_pos, attn_mask, cond_mask=None):
        """Decode the P in-flight frames; temporal attends [cached ctx ‖ fresh in-flight]. No frame
        mask (the sink is in the cache). Returns (x, s, k_unroped, v) — the in-flight temporal K/V."""
        xa, sa = self.spatial(x, s)
        x = x + self.gate_spatial * (xa - x)
        s = s + self.gate_spatial.view(1, -1, 1, 1) * (sa - s)
        xt, st, k_u, v = self.temporal.stream_read(x, s, ctx_k, ctx_v, ctx_pos, q_pos, attn_mask)
        x = x + self.gate_temporal * (xt - x)
        s = s + self.gate_temporal.view(1, -1, 1, 1) * (st - s)
        x = self._adaln_mlp(x, emb, cond, cond_mask=cond_mask, tok_emb_mask=None)
        s_in = rearrange(s, "b c s t -> b c t s 1")
        s = rearrange(self._adaln_mlp(s_in, emb, cond, cond_mask=cond_mask, tok_emb_mask=None),
                      "b c t s 1 -> b c s t")
        return x, s, k_u, v


class _DiTBlockFullWithState(_DiTBlockWithState):
    """Block whose attention is ONE joint spatio-temporal (frame-block-causal) pass; the AdaLN-Zero
    SwiGLU MLP, FiLM masks and gated residual are inherited. ``gate_attn`` is zero-init (identity at init)."""

    def __init__(self, dim: int, emb_dim: int, cond_dim: int, full_attn_ws, mlp_ratio: float = 8 / 3,
                 mlp_hidden_multiple: int = 8):
        nn.Module.__init__(self)
        self.full = full_attn_ws(dim)
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = _SwiGLU(dim, mlp_ratio=mlp_ratio, hidden_multiple=mlp_hidden_multiple)
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim * 2))
        self.cond_emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))
        self.gate_attn = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.gate_mlp = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        nn.init.zeros_(self.emb_layers[-1].weight); nn.init.zeros_(self.emb_layers[-1].bias)
        nn.init.zeros_(self.cond_emb_layers[-1].weight); nn.init.zeros_(self.cond_emb_layers[-1].bias)

    def forward(self, x, s, emb, cond, attn_mask=None, rope_pos=None, cond_mask=None,
                frame_emb_mask=None, full_bm=None):
        xa, sa = self.full(x, s, attn_mask=attn_mask, rope_pos=rope_pos, block_mask=full_bm)
        x = x + self.gate_attn * (xa - x)
        s = s + self.gate_attn.view(1, -1, 1, 1) * (sa - s)
        x = self._adaln_mlp(x, emb, cond, cond_mask=cond_mask, tok_emb_mask=frame_emb_mask)
        s_in = rearrange(s, "b c s t -> b c t s 1")
        s = rearrange(self._adaln_mlp(s_in, emb, cond, cond_mask=cond_mask,
                                      tok_emb_mask=frame_emb_mask), "b c t s 1 -> b c s t")
        return x, s

    def forward_prefill(self, x, s, emb, cond, attn_mask=None, rope_pos=None, cond_mask=None,
                        frame_emb_mask=None, full_bm=None):
        """Run over CONTEXT frames AND capture the UN-ROPED joint K/V, (B,heads,frames*tpf,d), from
        the block's own pre-norm input."""
        k_u, v = self.full.stream_encode(x, s)
        xa, sa = self.full(x, s, attn_mask=attn_mask, rope_pos=rope_pos, block_mask=full_bm)
        x = x + self.gate_attn * (xa - x)
        s = s + self.gate_attn.view(1, -1, 1, 1) * (sa - s)
        x = self._adaln_mlp(x, emb, cond, cond_mask=cond_mask, tok_emb_mask=frame_emb_mask)
        s_in = rearrange(s, "b c s t -> b c t s 1")
        s = rearrange(self._adaln_mlp(s_in, emb, cond, cond_mask=cond_mask, tok_emb_mask=frame_emb_mask),
                      "b c t s 1 -> b c s t")
        return x, s, k_u, v

    def forward_stream(self, x, s, emb, cond, ctx_k, ctx_v, ctx_pos, q_pos, attn_mask,
                       cond_mask=None, full_bm=None):
        """Decode the P in-flight frames against the cached context; same signature/return as the
        factorized block so a mixed stack is driven uniformly."""
        xa, sa, k_u, v = self.full.stream_read(x, s, ctx_k, ctx_v, ctx_pos, q_pos, attn_mask,
                                              block_mask=full_bm)
        x = x + self.gate_attn * (xa - x)
        s = s + self.gate_attn.view(1, -1, 1, 1) * (sa - s)
        x = self._adaln_mlp(x, emb, cond, cond_mask=cond_mask, tok_emb_mask=None)
        s_in = rearrange(s, "b c s t -> b c t s 1")
        s = rearrange(self._adaln_mlp(s_in, emb, cond, cond_mask=cond_mask, tok_emb_mask=None),
                      "b c t s 1 -> b c s t")
        return x, s, k_u, v


class DiTLatentDynamics(nn.Module):
    """Latent dynamics DiT with a per-frame state token, learned streaming sink, separate memory
    encoder, detached contact readout and training-only MCP heads. Input/output grid is (B,C,T,H,W)."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        state_dim: int,
        spatial_grid: Tuple[int, int],
        dim: int = 512,
        action_emb_dim: int = 512,
        cond_emb_dim: int = 512,
        depth: int = 12,
        patch_size: int = 1,
        mlp_ratio: float = 8 / 3,
        mlp_hidden_multiple: int = 8,
        attn_dim_head: int = 64,
        attn_heads: int = 8,
        grad_checkpoint: bool = False,
        contact_dim: int = 2,
        contact_readout_layer: int = -1,
        full_attn_layers: Optional[Sequence[int]] = None,
        mcp_depths: int = 0,
        mcp_blocks_per_depth: int = 1,
        mcp_tap_layers: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.patch_size = patch_size
        self.spatial_grid = tuple(spatial_grid)
        self.state_dim = int(state_dim)
        self.contact_dim = int(contact_dim)
        # which block's OUTPUT state feature feeds the contact readout (0-indexed; -1 == middle block)
        self.contact_readout_layer = (depth // 2 if int(contact_readout_layer) < 0
                                      else int(contact_readout_layer))

        # conditioning embeddings: noise emb = concat(cur, stop) each cond_emb_dim//2; action -> action_emb_dim
        self.noise_level_pos_embedding = nn.Sequential(
            Timesteps(cond_emb_dim // 2, True, 0),
            TimestepEmbedding(in_channels=cond_emb_dim // 2, time_embed_dim=cond_emb_dim // 2),
        )
        self.action_emd = nn.Sequential(
            nn.Linear(action_dim, 128), nn.SiLU(),
            nn.Linear(128, action_emb_dim),
        )
        # cache_if_possible=False: the lib caches freqs by seq_len only and would ignore varying Δt positions
        self.rotary_time_pos_embedding = RotaryEmbedding(dim=attn_dim_head, cache_if_possible=False)
        self.patch_embed = nn.Conv3d(latent_dim, dim, kernel_size=(1, patch_size, patch_size),
                                     stride=(1, patch_size, patch_size))
        self.unpatch = nn.ConvTranspose3d(dim, latent_dim, kernel_size=(1, patch_size, patch_size),
                                          stride=(1, patch_size, patch_size))
        nn.init.zeros_(self.unpatch.weight); nn.init.zeros_(self.unpatch.bias)   # AdaLN-Zero output

        srope = (self.spatial_grid[0] // patch_size, self.spatial_grid[1] // patch_size)   # token grid
        spatial_ws = partial(SpatialAttentionBlockWithState, heads=attn_heads, dim_head=attn_dim_head,
                             spatial_rope_grid=srope)
        temporal_ws = partial(TemporalAttentionBlockWithState, heads=attn_heads, dim_head=attn_dim_head,
                              rotary_emb=self.rotary_time_pos_embedding)
        full_set = {int(i) % depth for i in full_attn_layers} if full_attn_layers is not None else set()
        self.full_attn_layers = sorted(full_set)
        full_ws = partial(FullAttentionBlockWithState, spatial_grid=srope, heads=attn_heads,
                          dim_head=attn_dim_head)
        self.blocks = nn.ModuleList([
            (_DiTBlockFullWithState(dim, cond_emb_dim, action_emb_dim, full_ws, mlp_ratio=mlp_ratio,
                                    mlp_hidden_multiple=mlp_hidden_multiple) if i in full_set else
             _DiTBlockWithState(dim, cond_emb_dim, action_emb_dim, spatial_ws, temporal_ws,
                                mlp_ratio=mlp_ratio, mlp_hidden_multiple=mlp_hidden_multiple))
            for i in range(depth)
        ])
        # state in/out projections; state_out zero-init (whole net predicts 0 at init)
        self.state_in = nn.Sequential(nn.Linear(self.state_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.state_out_norm = nn.RMSNorm(dim, eps=1e-6)
        self.state_out = nn.Linear(dim, self.state_dim)
        nn.init.zeros_(self.state_out.weight); nn.init.zeros_(self.state_out.bias)
        # detached contact readout: small MLP over the mid-layer state feature (no gradient into the trunk)
        self.contact_ro_norm = nn.RMSNorm(dim, eps=1e-6)
        self.contact_ro_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, self.contact_dim))
        # streaming sink: a learned frame (post-patchify patch bank + state vector), never noised / RoPEd / conditioned
        gh, gw = srope
        self.sink_patch = nn.Parameter(torch.randn(1, dim, 1, gh, gw) * 0.02)
        self.sink_state = nn.Parameter(torch.randn(1, dim, 1) * 0.02)
        # separate memory encoder: the leading n_mem frames use their OWN patch/state projectors
        self.mem_patch_embed = nn.Conv3d(latent_dim, dim, kernel_size=(1, patch_size, patch_size),
                                         stride=(1, patch_size, patch_size))
        self.mem_state_in = nn.Sequential(nn.Linear(self.state_dim, dim), nn.SiLU(),
                                          nn.Linear(dim, dim))
        self.out_norm = nn.RMSNorm(dim, eps=1e-6)

        # MCP heads (training-only; built LAST so the trunk's init is independent of them)
        self.mcp_depths = int(mcp_depths)
        self.mcp_tap_layers: List[int] = []
        self.mcp = None
        if self.mcp_depths > 0:
            self.mcp_tap_layers = (sorted({int(i) % depth for i in mcp_tap_layers})
                                   if mcp_tap_layers is not None
                                   else mcp_tap_layers_for_depth(depth))
            mk_mcp_block = (lambda: _DiTBlockWithState(dim, cond_emb_dim, action_emb_dim, spatial_ws,
                                                       temporal_ws, mlp_ratio=mlp_ratio,
                                                       mlp_hidden_multiple=mlp_hidden_multiple))
            self.mcp = McpHeads(dim, self.mcp_depths, int(mcp_blocks_per_depth),
                                len(self.mcp_tap_layers), mk_mcp_block)

    def _ac(self, blk, *args):
        """Activation-checkpoint one block (full recompute; positional args only)."""
        return checkpoint(blk, *args, use_reentrant=False)

    def init_mcp_from_backbone(self) -> bool:
        """Copy the trunk's LAST blocks into every MCP depth. Call AFTER any warm-start load."""
        if self.mcp is None:
            return False
        return self.mcp.init_from_backbone(self.blocks)

    def forward_mcp(self, taps_x, taps_s, mcp_inputs, *, n_ctx: int, attn_mask=None, cond_mask=None,
                    frame_emb_mask=None):
        """Chained Multi-Chunk Prediction -> [(v_img_k (T,B,C,H,W), v_state_k (T,B,pose)), ...].

        taps_x/taps_s: trunk features at ``mcp_tap_layers``, (B,dim,N,h',w') / (B,dim,1,N). Rows
        ``[:n_ctx]`` (sink+memory) keep the fused features at every depth; rows ``[n_ctx:]`` are the
        future rows the chain rewrites. mcp_inputs[k]: dict(x (T,B,C,H,W), state (T,B,pose),
        noise_levels (N,B) scaled, cond (N,B,A), rope_pos)."""
        fused_x = self.mcp.fuse(taps_x, 1)                               # (B,dim,N,h',w')
        fused_s = self.mcp.fuse(taps_s, 1)                               # (B,dim,1,N)
        ctx_x, prev_x = fused_x[:, :, :n_ctx], fused_x[:, :, n_ctx:]
        ctx_s, prev_s = fused_s[:, :, :, :n_ctx], fused_s[:, :, :, n_ctx:]
        outs = []
        for k, d in enumerate(mcp_inputs):
            nl = d["noise_levels"]
            n_emb = self.noise_level_pos_embedding(rearrange(nl, "f b -> b f"))
            s_emb = self.noise_level_pos_embedding(
                rearrange(d.get("stop_noise_levels", nl), "f b -> b f"))
            emb = torch.cat([n_emb, s_emb], dim=-1)
            cond = self.action_emd(rearrange(d["cond"], "t b a -> b t a"))
            ex = self.patch_embed(rearrange(d["x"], "t b c h w -> b c t h w"))
            es = rearrange(self.state_in(rearrange(d["state"], "t b p -> b t p")), "b t d -> b d t")
            x = torch.cat([ctx_x, self.mcp.project(k, prev_x, ex)], dim=2)
            s = torch.cat([ctx_s, self.mcp.project(k, prev_s, es[:, :, None])], dim=3)
            for blk in self.mcp.blocks[k]:
                if self.grad_checkpoint and self.training:
                    x, s = self._ac(blk, x, s, emb, cond, attn_mask, d["rope_pos"], cond_mask,
                                    frame_emb_mask)
                else:
                    x, s = blk(x, s, emb, cond, attn_mask=attn_mask, rope_pos=d["rope_pos"],
                               cond_mask=cond_mask, frame_emb_mask=frame_emb_mask)
            prev_x, prev_s = x[:, :, n_ctx:], s[:, :, :, n_ctx:]         # the causal chain
            xo = rearrange(self.out_norm(rearrange(x, "b c t h w -> b t h w c")),
                           "b t h w c -> b c t h w")
            v_img = rearrange(self.unpatch(xo), "b c t h w -> t b c h w")
            v_state = rearrange(
                self.state_out(self.state_out_norm(rearrange(s[:, :, 0], "b d t -> b t d"))),
                "b t p -> t b p")
            outs.append((v_img, v_state))
        return outs

    def forward(self, x, noise_levels, stop_noise_levels, external_cond, state, rope_pos,
                attn_mask=None, cond_mask=None, n_mem=0, n_sink=0, full_bm=None, mcp_inputs=None):
        """x: (B,C,T,H,W); noise_levels/stop: (T,B); external_cond: (T,B,A); state: (T,B,state_dim)
        (noised); rope_pos: (T,) or (B,T) per-frame Δt positions; attn_mask: (T,T) or (B,T,T) temporal
        mask (True=attend, causality folded in; None -> causal); cond_mask: (B,T) in {0,1} action-FiLM
        gate. Frame layout ``[sink(n_sink) | memory(n_mem) | rest]``; the sink's input slot is ignored.
        Returns (v_img (B,C,T,H,W), v_state (T,B,state_dim), contact_logits (T,B,contact_dim)[, mcp_outs])."""
        noise_emb = self.noise_level_pos_embedding(rearrange(noise_levels, "f b -> b f"))
        stop_emb = self.noise_level_pos_embedding(rearrange(stop_noise_levels, "f b -> b f"))
        emb = torch.cat([noise_emb, stop_emb], dim=-1)                 # (B,T,cond_emb_dim)
        cond = self.action_emd(rearrange(external_cond, "t b d -> b t d"))  # (B,T,action_emb)

        # patchify: [sink bank | mem_patch_embed | patch_embed] (Conv3d is per-frame, so slicing T is exact)
        B0 = x.shape[0]
        if n_sink > 0:
            xs = self.sink_patch.expand(B0, -1, n_sink, -1, -1).to(x.dtype)
            if n_mem > 0:
                xm = self.mem_patch_embed(x[:, :, n_sink:n_sink + n_mem])
                xr = self.patch_embed(x[:, :, n_sink + n_mem:])
                x = torch.cat([xs, xm, xr], dim=2)
            else:
                x = torch.cat([xs, self.patch_embed(x[:, :, n_sink:])], dim=2)
        elif n_mem > 0:
            xm = self.mem_patch_embed(x[:, :, :n_mem])                 # (B,dim,n_mem,h',w')
            xr = self.patch_embed(x[:, :, n_mem:])                     # (B,dim,T-n_mem,h',w')
            x = torch.cat([xm, xr], dim=2)                             # (B,dim,T,h',w')
        else:
            x = self.patch_embed(x)                                    # (B,dim,T,h',w')
        # the sink carries no noised content -> zero its noise-level FiLM (exact identity)
        frame_emb_mask = None
        if n_sink > 0:
            frame_emb_mask = torch.ones((1, 1, x.shape[2], 1, 1), device=x.device, dtype=x.dtype)
            frame_emb_mask[:, :, :n_sink] = 0.0

        # state token: embed the noised state -> (B,dim,T) track, sink slots from the learned sink_state
        s_in = rearrange(state, "t b p -> b t p")                      # (B,T,pose)
        if n_mem > 0:
            sm = self.mem_state_in(s_in[:, n_sink:n_sink + n_mem])     # (B,n_mem,dim)
            sr = self.state_in(s_in[:, n_sink + n_mem:])               # (B,T-n_sink-n_mem,dim)
            s = torch.cat([sm, sr], dim=1)                             # (B,T-n_sink,dim)
        else:
            s = self.state_in(s_in[:, n_sink:])                        # (B,T-n_sink,dim)
        s = rearrange(s, "b t d -> b d t")                             # (B,dim,T-n_sink)
        if n_sink > 0:
            s = torch.cat([self.sink_state.expand(s.shape[0], -1, n_sink).to(s.dtype), s], dim=2)
        side = s[:, :, None]                                           # (B,dim,1,T)
        ro_feat = None                                                 # captured mid-layer state feature
        # one flex BlockMask for all full blocks (create_block_mask graph-breaks dynamo; build it once)
        if full_bm is None and self.full_attn_layers:
            nf = x.shape[2]
            fmask = attn_mask if attn_mask is not None else torch.ones(
                (nf, nf), dtype=torch.bool, device=x.device).tril()
            full_bm = self.make_full_block_mask(fmask, x.shape[0], nf, nf, x.device)
        mcp_on = mcp_inputs is not None and self.mcp is not None
        taps_x, taps_s = [], []
        for bi, blk in enumerate(self.blocks):
            if self.grad_checkpoint and self.training:
                # full blocks take full_bm as an extra POSITIONAL arg (checkpoint forbids kwargs)
                extra = (full_bm,) if isinstance(blk, _DiTBlockFullWithState) else ()
                x, side = self._ac(blk, x, side, emb, cond, attn_mask, rope_pos, cond_mask,
                                   frame_emb_mask, *extra)
            elif isinstance(blk, _DiTBlockFullWithState):
                x, side = blk(x, side, emb, cond, attn_mask=attn_mask, rope_pos=rope_pos,
                              cond_mask=cond_mask, frame_emb_mask=frame_emb_mask, full_bm=full_bm)
            else:
                x, side = blk(x, side, emb, cond, attn_mask=attn_mask, rope_pos=rope_pos,
                              cond_mask=cond_mask, frame_emb_mask=frame_emb_mask)
            if bi == self.contact_readout_layer:
                # DETACHED: the contact MLP trains on it, no gradient reaches the trunk
                ro_feat = side[:, :, 0].detach()                       # (B,dim,T)
            if mcp_on and bi in self.mcp_tap_layers:
                taps_x.append(x); taps_s.append(side)                  # LIVE features (grad reaches the trunk)
        x = rearrange(self.out_norm(rearrange(x, "b c t h w -> b t h w c")), "b t h w c -> b c t h w")
        v_img = self.unpatch(x)                                        # (B,latent_dim,T,H,W)
        s_out = self.state_out_norm(rearrange(side[:, :, 0], "b d t -> b t d"))
        v_state = rearrange(self.state_out(s_out), "b t p -> t b p")   # (T,B,state_dim)
        mcp = ()
        if mcp_on:
            assert len(taps_x) == len(self.mcp_tap_layers), "MCP did not collect every tap layer"
            mcp = (self.forward_mcp(taps_x, taps_s, mcp_inputs, n_ctx=n_sink + n_mem,
                                    attn_mask=attn_mask, cond_mask=cond_mask,
                                    frame_emb_mask=frame_emb_mask),)
        c_in = self.contact_ro_norm(rearrange(ro_feat, "b d t -> b t d"))       # (B,T,dim) detached
        contact_logits = rearrange(self.contact_ro_mlp(c_in), "b t k -> t b k")  # (T,B,contact_dim)
        return (v_img, v_state, contact_logits) + mcp

    # ================= STREAMING KV CACHE: prefill (build cache) + decode (in-flight) =============
    # Cache = per-block (k_unroped, v) packed to temporal rows. prefill runs the CONTEXT frames
    # [sink|mem|committed] and captures each block's un-roped K/V; decode forwards ONLY the P in-flight
    # frames and attends against the cache re-roped to the frames' CURRENT positions.
    def make_full_block_mask(self, frame_mask, B: int, n_q_frames: int, n_kv_frames: int, device=None):
        """Block-sparse mask for the full-attention layers (identical for all of them). Build it
        OUTSIDE a compiled region and pass it as ``full_bm``."""
        for blk in self.blocks:
            if isinstance(blk, _DiTBlockFullWithState):
                return blk.full.make_block_mask(frame_mask, B, n_q_frames, n_kv_frames, device)
        return None

    def _side_from_state(self, state, n_lead: int, n_mem: int):
        """State track for `state` (T',B,pose) as in forward: mem_state_in on the leading n_mem frames,
        state_in on the rest, `n_lead` sink slots prepended. Returns the side stream (B,dim,1,T')."""
        s_in = rearrange(state, "t b p -> b t p")
        if n_mem > 0:
            sm = self.mem_state_in(s_in[:, n_lead:n_lead + n_mem])
            sr = self.state_in(s_in[:, n_lead + n_mem:])
            s = torch.cat([sm, sr], dim=1)
        else:
            s = self.state_in(s_in[:, n_lead:])
        s = rearrange(s, "b t d -> b d t")
        if n_lead > 0:
            s = torch.cat([self.sink_state.expand(s.shape[0], -1, n_lead).to(s.dtype), s], dim=2)
        return s[:, :, None]

    def stream_prefill(self, x, noise_levels, external_cond, state, rope_pos, attn_mask,
                       cond_mask, n_sink, n_mem, full_bm=None):
        """Run the CONTEXT frames [sink|mem|committed] and RETURN the per-block cache [(k_unroped,v), …]."""
        emb = torch.cat([self.noise_level_pos_embedding(rearrange(noise_levels, "f b -> b f")),
                         self.noise_level_pos_embedding(rearrange(noise_levels, "f b -> b f"))], dim=-1)
        cond = self.action_emd(rearrange(external_cond, "t b d -> b t d"))
        B0 = x.shape[0]
        parts = []
        if n_sink > 0:
            parts.append(self.sink_patch.expand(B0, -1, n_sink, -1, -1).to(x.dtype))
        if n_mem > 0:
            parts.append(self.mem_patch_embed(x[:, :, n_sink:n_sink + n_mem]))
            if x.shape[2] > n_sink + n_mem:                          # committed c frames (may be empty)
                parts.append(self.patch_embed(x[:, :, n_sink + n_mem:]))
        elif x.shape[2] > n_sink:
            parts.append(self.patch_embed(x[:, :, n_sink:]))
        x = torch.cat(parts, dim=2)
        frame_emb_mask = None
        if n_sink > 0:
            frame_emb_mask = torch.ones((1, 1, x.shape[2], 1, 1), device=x.device, dtype=x.dtype)
            frame_emb_mask[:, :, :n_sink] = 0.0
        side = self._side_from_state(state, n_sink, n_mem)
        cache = []
        for blk in self.blocks:
            kw = {"full_bm": full_bm} if isinstance(blk, _DiTBlockFullWithState) else {}
            x, side, k_u, v = blk.forward_prefill(x, side, emb, cond, attn_mask=attn_mask,
                                                  rope_pos=rope_pos, cond_mask=cond_mask,
                                                  frame_emb_mask=frame_emb_mask, **kw)
            cache.append((k_u, v))
        return cache

    def stream_decode(self, x, noise_levels, external_cond, state, cache, ctx_pos, q_pos, attn_mask,
                      cond_mask, full_bm=None):
        """Forward ONLY the P in-flight frames (main encoders, no sink/mem), attending each block
        against cache[bi] re-roped to `ctx_pos`. Returns (v_img (B,latent,P,H,W), v_state (P,B,pose),
        contact (P,B,cdim), inflight_kv [(k_u,v),…]) — the last for commit-promotion."""
        emb = torch.cat([self.noise_level_pos_embedding(rearrange(noise_levels, "f b -> b f")),
                         self.noise_level_pos_embedding(rearrange(noise_levels, "f b -> b f"))], dim=-1)
        cond = self.action_emd(rearrange(external_cond, "t b d -> b t d"))
        x = self.patch_embed(x)                                        # in-flight are all `c` frames
        side = self._side_from_state(state, 0, 0)                      # state_in only, no sink/mem
        ro_feat, inflight_kv = None, []
        for bi, blk in enumerate(self.blocks):
            ck, cv = cache[bi]
            kw = {"full_bm": full_bm} if isinstance(blk, _DiTBlockFullWithState) else {}
            x, side, k_u, v = blk.forward_stream(x, side, emb, cond, ck, cv, ctx_pos, q_pos,
                                                 attn_mask, cond_mask=cond_mask, **kw)
            inflight_kv.append((k_u, v))
            if bi == self.contact_readout_layer:
                ro_feat = side[:, :, 0].detach()
        x = rearrange(self.out_norm(rearrange(x, "b c t h w -> b t h w c")), "b t h w c -> b c t h w")
        v_img = self.unpatch(x)
        v_state = rearrange(self.state_out(self.state_out_norm(rearrange(side[:, :, 0], "b d t -> b t d"))),
                            "b t p -> t b p")
        contact = rearrange(self.contact_ro_mlp(self.contact_ro_norm(
            rearrange(ro_feat, "b d t -> b t d"))), "b t k -> t b k")
        return v_img, v_state, contact, inflight_kv
