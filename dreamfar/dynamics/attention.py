"""Attention blocks for the DiT latent dynamics: factorized spatial (2D pixel RoPE) and temporal
(Δt-RoPE, causal) attention over [H*W patch tokens | 1 state token], the joint spatio-temporal
flex-attention block (axial 3D RoPE), and their streaming KV-cache (un-roped store, re-rope at gather).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Ordered SDPA preference: flash (bf16/fp16 only) -> mem-efficient -> math (always available, fp32 ok).
_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


# ---- 2D axial "pixel" RoPE for the spatial attention ------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope_mix(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply precomputed cos/sin RoPE. cos/sin: (1,1,N,D) broadcast over (B,H,N,D)."""
    return x * cos + _rotate_half(x) * sin


def _pixel_rope_2d(grid_hw: Tuple[int, int], dim_head: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """2D PIXEL RoPE table for an H×W grid in raster order: (cos, sin) each (H*W, dim_head).
    dim_head is split across the two axes; each axis gets linear ("pixel") frequencies."""
    H, W = grid_hw
    assert dim_head % 4 == 0, f"dim_head={dim_head} must be divisible by 2*ndim=4 for 2D RoPE"
    dim_per_axis = dim_head // 2      # half the head dim per spatial axis
    half = dim_per_axis // 2          # cos/sin pairs per axis
    hh, ww = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cos_parts, sin_parts = [], []
    for pos, size in ((hh.reshape(-1).float(), H), (ww.reshape(-1).float(), W)):
        freqs = torch.linspace(1.0, size / 2.0, half)          # PIXEL: linear freqs
        angles = torch.outer(pos, freqs)                       # (H*W, half)
        cos_parts.append(angles.cos()); sin_parts.append(angles.sin())
    cos_half = torch.cat(cos_parts, dim=-1)                    # (H*W, dim_head//2)
    sin_half = torch.cat(sin_parts, dim=-1)
    cos = cos_half.repeat(1, 2)                                # (H*W, dim_head)
    sin = sin_half.repeat(1, 2)
    return cos, sin


class Attention(nn.Module):
    """Multi-head SDPA with per-head learned RMSNorm on q,k. Position signal is ONE of: a temporal
    ``rotary_emb`` (Δt-RoPE at the caller's positions), a ``spatial_rope_grid`` (2D pixel RoPE table,
    identity rows for ``spatial_extra_tokens`` appended side tokens), or neither (the joint block
    applies its own axial 3D RoPE through ``full_attn``)."""

    def __init__(
        self,
        query_dim: int,
        heads: int = 4,
        dim_head: int = 32,
        rotary_emb: Optional[RotaryEmbedding] = None,
        spatial_rope_grid: Optional[Tuple[int, int]] = None,
        spatial_extra_tokens: int = 0,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.rotary_emb = rotary_emb
        self.spatial_rope_grid = spatial_rope_grid
        self.spatial_extra_tokens = int(spatial_extra_tokens)
        if spatial_rope_grid is not None:
            assert rotary_emb is None, "spatial_rope_grid and temporal rotary_emb are exclusive"
            cos, sin = _pixel_rope_2d(spatial_rope_grid, dim_head)
            if self.spatial_extra_tokens > 0:
                pad = self.spatial_extra_tokens
                cos = torch.cat([cos, torch.ones(pad, dim_head)], dim=0)    # identity rotation
                sin = torch.cat([sin, torch.zeros(pad, dim_head)], dim=0)
            self.register_buffer("_srope_cos", cos, persistent=False)
            self.register_buffer("_srope_sin", sin, persistent=False)
        self.to_qkv = nn.Linear(query_dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, query_dim)
        # per-head learned RMSNorm on q,k over dim_head; SDPA keeps its default 1/sqrt(d) scale
        self.q_norm = nn.RMSNorm(dim_head, eps=1e-6)
        self.k_norm = nn.RMSNorm(dim_head, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, is_causal: bool = False,
                attn_mask: Optional[torch.Tensor] = None,
                rope_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attn_mask (…,N_q,N_k) bool REPLACES is_causal (the caller folds causality in). rope_pos
        # (temporal only): TRUE per-frame Δt positions, (N,) shared or (rows,N) per-row.
        q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))

        if self.rotary_emb is not None:
            rp = rope_pos.to(q.device)
            if rp.dim() == 1:
                freqs = self.rotary_emb.forward(rp, seq_len=q.shape[2])              # (N,d)
            else:
                rows, N = rp.shape
                freqs = self.rotary_emb.forward(rp.reshape(-1), seq_len=rows * N).reshape(rows, N, -1)
                freqs = freqs.unsqueeze(1)                                           # (rows,1,N,d)
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        else:
            # 2D pixel RoPE on the (H*W [+ side]) spatial tokens; applied to q,k only
            cos = self._srope_cos.to(q.dtype)[None, None]
            sin = self._srope_sin.to(q.dtype)[None, None]
            q = _rope_mix(q, cos, sin)
            k = _rope_mix(k, cos, sin)

        return self._sdpa_out(q, k, v, is_causal, attn_mask=attn_mask)

    def _sdpa_out(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool,
                  attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Shared SDPA tail: qk-norm (post-rotary, per vector -> cache-concat safe) + backend + out proj."""
        q = self.q_norm(q)
        k = self.k_norm(k)

        if attn_mask is not None:
            is_causal = False
            attn_mask = attn_mask.to(q.device)
            # (Nq,Nk) broadcasts as-is; a PER-ROW (rows,Nq,Nk) mask needs a head axis.
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
        if q.shape[0] >= 65536:
            # MATH-only above a huge batch dim avoids a known flash/mem-efficient CUDA error.
            backends = [SDPBackend.MATH]
            q, k, v = (t.contiguous() for t in (q, k, v))
        elif attn_mask is not None:
            backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        else:
            backends = _SDPA_BACKENDS
        with sdpa_kernel(backends=backends):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, attn_mask=attn_mask)

        out = rearrange(out, "b h n d -> b n (h d)").to(q.dtype)
        return self.to_out(out)

    def full_attn(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                  block_mask) -> torch.Tensor:
        """JOINT spatio-temporal attention over a flattened (frames x tokens) sequence with a
        precomputed axial 3D RoPE (cos,sin) table and a FlexAttention block mask."""
        q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        # cast the table to q's dtype (under autocast the normed input is fp32 while to_qkv emits bf16)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = _rope_mix(q, cos, sin)
        k = _rope_mix(k, cos, sin)
        q, k = self.q_norm(q), self.k_norm(k)
        out = _flex_fn()(q, k, v, block_mask=block_mask)
        out = rearrange(out, "b h n d -> b n (h d)").to(q.dtype)
        return self.to_out(out)

    # ---- streaming KV cache for FULL (joint) attention: un-roped store, axial re-rope at gather ----
    def full_stream_encode(self, hidden_states: torch.Tensor):
        """UN-ROPED K/V for the frames to cache. (B,n,c) -> (B,heads,n,d) each, n = frames*tokens."""
        _, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        return k.detach(), v.detach()

    def full_stream_read(self, hidden_states: torch.Tensor, ctx_k_unroped: torch.Tensor,
                         ctx_v: torch.Tensor, ctx_cos: torch.Tensor, ctx_sin: torch.Tensor,
                         q_cos: torch.Tensor, q_sin: torch.Tensor, block_mask):
        """In-flight tokens attend [re-roped cached ctx ‖ fresh in-flight]. Returns
        (out, k_unroped, v) — the last two are this call's in-flight cache entries."""
        q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        k_store, v_store = k.detach(), v.detach()                  # pre-rope: what the cache keeps
        qc, qs = q_cos.to(q.dtype), q_sin.to(q.dtype)
        q = _rope_mix(q, qc, qs)
        k = _rope_mix(k, qc, qs)
        ck = _rope_mix(ctx_k_unroped.to(q.dtype), ctx_cos.to(q.dtype), ctx_sin.to(q.dtype))
        K = torch.cat([ck, k], dim=2)
        V = torch.cat([ctx_v.to(v.dtype), v], dim=2)
        q, K = self.q_norm(q), self.k_norm(K)                      # per-vector => concat-safe
        out = _flex_fn()(q, K, V, block_mask=block_mask)
        out = rearrange(out, "b h n d -> b n (h d)").to(q.dtype)
        return self.to_out(out), k_store, v_store

    # ---- streaming KV cache (temporal attention): un-roped store + re-rope at gather ---------------
    def _freqs(self, pos):
        """Rotary freqs for positions `pos` ((N,) shared or (rows,N) per-row) -> (N,d) or (rows,1,N,d).
        No `seq_len`: the lib would otherwise serve cached freqs BY INDEX and ignore the shifted values."""
        if pos.dim() == 1:
            return self.rotary_emb.forward(pos)                                    # (N,d)
        rows, N = pos.shape
        f = self.rotary_emb.forward(pos.reshape(-1)).reshape(rows, N, -1)
        return f.unsqueeze(1)                                                      # (rows,1,N,d)

    def stream_encode(self, hidden_states: torch.Tensor):
        """UN-ROPED, un-normed K/V for the frames to cache. (rows,n,c) -> (rows,heads,n,d) each."""
        _, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        return k.detach(), v.detach()

    def stream_read(self, hidden_states: torch.Tensor, ctx_k_unroped: torch.Tensor,
                    ctx_v: torch.Tensor, ctx_pos: torch.Tensor, q_pos: torch.Tensor,
                    attn_mask: torch.Tensor):
        """In-flight queries attend [re-roped cached ctx ‖ fresh in-flight]. Returns
        (out (rows,P,c), k_unroped (rows,heads,P,d), v (rows,heads,P,d)). ctx_pos/q_pos: (N,) or
        (rows,N). attn_mask: (rows,P,n_ctx+P) or (P,n_ctx+P) bool (True=attend)."""
        q, k, v = self.to_qkv(hidden_states).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        k_unroped, v_out = k.detach(), v.detach()                       # for promotion on commit
        ck = apply_rotary_emb(self._freqs(ctx_pos.to(q.device)), ctx_k_unroped)
        qf = self._freqs(q_pos.to(q.device))
        q = apply_rotary_emb(qf, q)
        k = apply_rotary_emb(qf, k)
        K = torch.cat([ck, k], dim=2)
        V = torch.cat([ctx_v, v], dim=2)
        out = self._sdpa_out(q, K, V, is_causal=False, attn_mask=attn_mask)
        return out, k_unroped, v_out


class AttentionBlock(nn.Module):
    """Pre-norm (RMSNorm) residual attention block."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        rotary_emb: Optional[RotaryEmbedding] = None,
        spatial_rope_grid: Optional[Tuple[int, int]] = None,
        spatial_extra_tokens: int = 0,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.attn = Attention(query_dim=dim, heads=heads, dim_head=dim_head, rotary_emb=rotary_emb,
                              spatial_rope_grid=spatial_rope_grid,
                              spatial_extra_tokens=spatial_extra_tokens)

    def forward(self, x: torch.Tensor, is_causal: bool = False,
                attn_mask: Optional[torch.Tensor] = None,
                rope_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x + self.attn(self.norm(x), is_causal=is_causal, attn_mask=attn_mask, rope_pos=rope_pos)

    def stream_encode(self, x: torch.Tensor):
        """Un-roped K/V from the NORMED hidden states (where forward feeds `self.attn`)."""
        return self.attn.stream_encode(self.norm(x))

    def stream_read(self, x: torch.Tensor, ctx_k_unroped, ctx_v, ctx_pos, q_pos, attn_mask):
        """In-flight residual: x + attn.stream_read(norm(x), cached ctx…). Returns (out, k_unroped, v)."""
        out, k_u, v = self.attn.stream_read(self.norm(x), ctx_k_unroped, ctx_v, ctx_pos, q_pos, attn_mask)
        return x + out, k_u, v


class _TemporalAttentionBlock(nn.Module):
    """Causal attention over the time axis with rotary (Δt) positions."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32,
                 rotary_emb: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.attn_block = AttentionBlock(dim, heads, dim_head, rotary_emb=rotary_emb)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                rope_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # a given attn_mask already folds causality in (SDPA takes one or the other)
        return self.attn_block(x, is_causal=attn_mask is None, attn_mask=attn_mask, rope_pos=rope_pos)

    def stream_encode(self, x):
        return self.attn_block.stream_encode(x)

    def stream_read(self, x, ctx_k_unroped, ctx_v, ctx_pos, q_pos, attn_mask):
        return self.attn_block.stream_read(x, ctx_k_unroped, ctx_v, ctx_pos, q_pos, attn_mask)


# ---- state-token-aware (B,C,T,H,W) + (B,C,S,T) variants, S = 1 side token (the state token) --------
class SpatialAttentionBlockWithState(nn.Module):
    """Spatial self-attention over [H*W patch tokens ‖ S side tokens], per frame. The side token gets
    an identity row in the 2D RoPE table (``spatial_extra_tokens=1``)."""

    def __init__(self, dim: int, **kw):
        super().__init__()
        self.block = AttentionBlock(dim, spatial_extra_tokens=1, **kw)

    def forward(self, x_img: torch.Tensor, x_side: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = x_img.shape
        S = x_side.shape[2]
        img = rearrange(x_img, "b c t h w -> (b t) (h w) c")
        side = rearrange(x_side, "b c s t -> (b t) s c")
        tokens = torch.cat([img, side], dim=1)                     # (B*T, H*W+S, C)
        out = self.block(tokens, is_causal=False)                  # spatial: never causal
        img_o, side_o = out[:, :-S], out[:, -S:]
        x_img = rearrange(img_o, "(b t) (h w) c -> b c t h w", b=B, t=T, h=H, w=W)
        x_side = rearrange(side_o, "(b t) s c -> b c s t", b=B, t=T)
        return x_img, x_side


class TemporalAttentionBlockWithState(nn.Module):
    """Temporal self-attention: each (h,w) patch track and each side token track attends over T with
    the SAME block. Rows are packed [img: (b h w) | side: (b s)]."""

    def __init__(self, dim: int, **kw):
        super().__init__()
        self.block = _TemporalAttentionBlock(dim, **kw)

    def forward(self, x_img: torch.Tensor, x_side: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                rope_pos: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T, H, W = x_img.shape
        S = x_side.shape[2]
        img = rearrange(x_img, "b c t h w -> (b h w) t c")
        side = rearrange(x_side, "b c s t -> (b s) t c")           # S tracks per batch elem
        # per-sample rope_pos (B,T) / attn_mask (B,T,T) are expanded to the packed row order
        rp = rope_pos
        if rp is not None and rp.dim() == 2:
            rp = torch.cat([rp.repeat_interleave(H * W, dim=0),
                            rp.repeat_interleave(S, dim=0)], dim=0)   # (B*H*W + B*S, T)
        am = attn_mask
        if am is not None and am.dim() == 3:
            am = torch.cat([am.repeat_interleave(H * W, dim=0),
                            am.repeat_interleave(S, dim=0)], dim=0)   # (B*H*W + B*S, T, T)
        seq = torch.cat([img, side], dim=0)
        out = self.block(seq, attn_mask=am, rope_pos=rp)
        n_img = B * H * W
        img_o, side_o = out[:n_img], out[n_img:]
        x_img = rearrange(img_o, "(b h w) t c -> b c t h w", b=B, h=H, w=W)
        x_side = rearrange(side_o, "(b s) t c -> b c s t", b=B, s=S)
        return x_img, x_side

    # ---- streaming KV cache: pack rows exactly like forward, cache/read per packed row -----------
    @staticmethod
    def _pack_rows(x_img, x_side):
        B, C, T, H, W = x_img.shape
        S = x_side.shape[2]
        img = rearrange(x_img, "b c t h w -> (b h w) t c")
        side = rearrange(x_side, "b c s t -> (b s) t c")
        return torch.cat([img, side], dim=0), B, H, W, S

    @staticmethod
    def _pack_perrow(t, per_sample_ndim, B, H, W, S):
        """Expand a PER-SAMPLE tensor (leading dim B) to packed rows like forward; a SHARED tensor
        (lower ndim) passes through and broadcasts over rows."""
        if t is None or t.dim() != per_sample_ndim:
            return t                                           # shared -> broadcast
        return torch.cat([t.repeat_interleave(H * W, dim=0), t.repeat_interleave(S, dim=0)], dim=0)

    def stream_encode(self, x_img, x_side):
        """Un-roped K/V for the frames in (x_img,x_side), packed to rows. (rows,heads,n,d) each."""
        seq, *_ = self._pack_rows(x_img, x_side)
        return self.block.stream_encode(seq)

    def stream_read(self, x_img, x_side, ctx_k_unroped, ctx_v, ctx_pos, q_pos, attn_mask):
        """In-flight (x_img,x_side) attend cached ctx. Returns (x_img_out, x_side_out, k_unroped, v)."""
        seq, B, H, W, S = self._pack_rows(x_img, x_side)
        cp = self._pack_perrow(ctx_pos, 2, B, H, W, S)         # per-sample pos is (B, n_ctx)
        qp = self._pack_perrow(q_pos, 2, B, H, W, S)           # per-sample pos is (B, P)
        am = self._pack_perrow(attn_mask, 3, B, H, W, S)       # per-sample mask is (B, P, n_ctx+P)
        out, k_u, v = self.block.stream_read(seq, ctx_k_unroped, ctx_v, cp, qp, am)
        n_img = B * H * W
        x_img_o = rearrange(out[:n_img], "(b h w) t c -> b c t h w", b=B, h=H, w=W)
        x_side_o = rearrange(out[n_img:], "(b s) t c -> b c s t", b=B, s=S)
        return x_img_o, x_side_o, k_u, v


# ======================= FULL (joint spatio-temporal) attention ====================================
_FLEX_FN = None


def _flex_fn():
    """Lazily import + compile flex_attention."""
    global _FLEX_FN
    if _FLEX_FN is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX_FN = torch.compile(flex_attention, dynamic=False)
    return _FLEX_FN


def _axial_dims(dim_head: int) -> Tuple[int, int, int]:
    """Split dim_head//2 rotary pairs across (temporal, h, w): temporal gets half, h/w share the rest."""
    half = dim_head // 2
    assert half % 2 == 0, f"dim_head={dim_head} must be a multiple of 4 for axial 3D RoPE"
    ht = half // 2
    hh = (half - ht) // 2
    hw = half - ht - hh
    return ht, hh, hw


def _spatial_axial_halves(grid_hw: Tuple[int, int], n_side: int, hh: int, hw: int):
    """Fixed per-frame spatial rotary halves for [H*W raster patches ‖ n_side side tokens]; side
    tokens get the identity rotation. Returns (cos,sin) each (H*W+n_side, hh+hw)."""
    H, W = grid_hw
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cos_p, sin_p = [], []
    for pos, size, n in ((yy.reshape(-1).float(), H, hh), (xx.reshape(-1).float(), W, hw)):
        freqs = torch.linspace(1.0, size / 2.0, n)              # PIXEL linear freqs (as _pixel_rope_2d)
        ang = torch.outer(pos, freqs)
        cos_p.append(ang.cos()); sin_p.append(ang.sin())
    cos = torch.cat(cos_p, dim=-1)                              # (H*W, hh+hw)
    sin = torch.cat(sin_p, dim=-1)
    if n_side > 0:
        cos = torch.cat([cos, torch.ones(n_side, hh + hw)], dim=0)
        sin = torch.cat([sin, torch.zeros(n_side, hh + hw)], dim=0)
    return cos, sin


def build_axial_rope_3d(pos_t: torch.Tensor, sp_cos: torch.Tensor, sp_sin: torch.Tensor,
                        ht: int, theta: float = 10000.0):
    """Per-TOKEN (cos,sin) for the frame-major flattened sequence. pos_t: (F,) shared or (B,F)
    per-sample temporal positions; sp_cos/sp_sin: (tpf, hh+hw). Returns (F*tpf, dim_head) or
    (B, F*tpf, dim_head) each, in the half-rotation layout ``_rope_mix`` expects."""
    tpf = sp_cos.shape[0]
    inv = theta ** (-torch.arange(ht, device=pos_t.device, dtype=torch.float32) / ht)
    ang = pos_t.float().unsqueeze(-1) * inv                     # (...,F,ht)
    tc, ts = ang.cos(), ang.sin()
    tc = tc.unsqueeze(-2).expand(*tc.shape[:-1], tpf, ht)       # (...,F,tpf,ht)
    ts = ts.unsqueeze(-2).expand(*ts.shape[:-1], tpf, ht)
    sc = sp_cos.to(tc.device).expand(*tc.shape[:-2], tpf, sp_cos.shape[-1])
    ss = sp_sin.to(ts.device).expand(*ts.shape[:-2], tpf, sp_sin.shape[-1])
    cos_h = torch.cat([tc, sc], dim=-1).flatten(-3, -2)         # (...,F*tpf,half)
    sin_h = torch.cat([ts, ss], dim=-1).flatten(-3, -2)
    return torch.cat([cos_h, cos_h], dim=-1), torch.cat([sin_h, sin_h], dim=-1)


class FullAttentionBlockWithState(nn.Module):
    """Pre-norm residual JOINT spatio-temporal attention over [frames x (H*W patches ‖ 1 side token)],
    frame-major, with a frame-block-causal FlexAttention mask and axial 3D RoPE. Returns post-residual
    (x_img, x_side) like the factorized WithState wrappers."""

    def __init__(self, dim: int, spatial_grid: Tuple[int, int], heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.grid = tuple(spatial_grid)
        self.tpf = self.grid[0] * self.grid[1] + 1                    # tokens per frame
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.attn = Attention(query_dim=dim, heads=heads, dim_head=dim_head)
        ht, hh, hw = _axial_dims(dim_head)
        self.ht = ht
        sc, ss = _spatial_axial_halves(self.grid, 1, hh, hw)
        self.register_buffer("_sp_cos", sc, persistent=False)
        self.register_buffer("_sp_sin", ss, persistent=False)

    def _block_mask(self, frame_mask: torch.Tensor, B: int, q_len: int, kv_len: int, device):
        """Block-sparse mask from the (Bm,Fq,Fkv) FRAME mask — nothing token-level is materialized."""
        from torch.nn.attention.flex_attention import create_block_mask
        tpf, fm = self.tpf, frame_mask
        shared = fm.shape[0] == 1

        def mask_mod(b, h, q_idx, kv_idx):
            bi = torch.zeros_like(b) if shared else b
            return fm[bi, q_idx // tpf, kv_idx // tpf]

        return create_block_mask(mask_mod, B, None, q_len, kv_len, device=device, _compile=True)

    def make_block_mask(self, frame_mask: torch.Tensor, B: int, n_q_frames: int, n_kv_frames: int,
                        device=None):
        """Build the block-sparse mask OUTSIDE a compiled region (create_block_mask graph-breaks
        dynamo); identical for every full block in a call, so build once and pass it down."""
        dev = device if device is not None else frame_mask.device
        fm = frame_mask.to(device=dev, dtype=torch.bool)
        if fm.dim() == 2:
            fm = fm[None]
        return self._block_mask(fm, B, n_q_frames * self.tpf, n_kv_frames * self.tpf, dev)

    # ---- streaming KV cache (frame-major, all tpf tokens per frame) ------------------------------
    def _pack(self, x_img, x_side):
        B, C, T, H, W = x_img.shape
        img = rearrange(x_img, "b c t h w -> b t (h w) c")
        side = rearrange(x_side, "b c s t -> b t s c")
        return torch.cat([img, side], dim=2).flatten(1, 2), B, T, H, W, x_side.shape[2]

    def _unpack(self, tok, B, T, H, W, S):
        o = tok.view(B, T, self.tpf, tok.shape[-1])
        return (rearrange(o[:, :, :H * W], "b t (h w) c -> b c t h w", h=H, w=W),
                rearrange(o[:, :, H * W:], "b t s c -> b c s t"))

    def _rope_for(self, pos, device, dtype):
        cos, sin = build_axial_rope_3d(pos.to(device), self._sp_cos, self._sp_sin, self.ht)
        cos, sin = cos.to(dtype), sin.to(dtype)
        return (cos[None, None] if cos.dim() == 2 else cos[:, None],
                sin[None, None] if sin.dim() == 2 else sin[:, None])

    def stream_encode(self, x_img, x_side):
        """UN-ROPED K/V for these frames from the NORMED hidden states: (k,v) each (B,heads,frames*tpf,d)."""
        tok, *_ = self._pack(x_img, x_side)
        return self.attn.full_stream_encode(self.norm(tok))

    def stream_read(self, x_img, x_side, ctx_k_unroped, ctx_v, ctx_pos, q_pos, attn_mask,
                    block_mask=None):
        """In-flight frames attend [cached ctx ‖ themselves]. ctx_pos/q_pos are per-FRAME temporal
        positions ((F,) shared or (B,F)); attn_mask is the per-FRAME (Bm,P,n_ctx+P) mask."""
        tok, B, T, H, W, S = self._pack(x_img, x_side)
        h = self.norm(tok)
        q_cos, q_sin = self._rope_for(q_pos, tok.device, h.dtype)
        c_cos, c_sin = self._rope_for(ctx_pos, tok.device, h.dtype)
        fm = attn_mask.to(device=tok.device, dtype=torch.bool)
        if fm.dim() == 2:
            fm = fm[None]
        q_len, kv_len = tok.shape[1], ctx_k_unroped.shape[2] + tok.shape[1]
        bm = (block_mask if block_mask is not None
              else self._block_mask(fm, B, q_len, kv_len, tok.device))
        out, k_u, v = self.attn.full_stream_read(h, ctx_k_unroped, ctx_v, c_cos, c_sin,
                                                q_cos, q_sin, block_mask=bm)
        xi, xs = self._unpack(tok + out, B, T, H, W, S)
        return xi, xs, k_u, v

    def forward(self, x_img: torch.Tensor, x_side: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                rope_pos: Optional[torch.Tensor] = None, block_mask=None):
        B, C, T, H, W = x_img.shape
        # frame-major flatten: (B, T*tpf, C)
        img = rearrange(x_img, "b c t h w -> b t (h w) c")
        side = rearrange(x_side, "b c s t -> b t s c")
        tok = torch.cat([img, side], dim=2).flatten(1, 2)
        N = tok.shape[1]
        h = self.norm(tok)
        # axial 3D RoPE at the caller's per-frame Δt positions
        cos, sin = build_axial_rope_3d(rope_pos.to(tok.device), self._sp_cos, self._sp_sin, self.ht)
        cos, sin = cos.to(h.dtype), sin.to(h.dtype)
        cos = cos[None, None] if cos.dim() == 2 else cos[:, None]     # -> (1|B, 1, N, d)
        sin = sin[None, None] if sin.dim() == 2 else sin[:, None]
        # frame-level mask (causality already folded in); None -> plain frame-causal
        fm = attn_mask
        if fm is None:
            fm = torch.ones(T, T, dtype=torch.bool, device=tok.device).tril()
        fm = fm.to(device=tok.device, dtype=torch.bool)
        if fm.dim() == 2:
            fm = fm[None]
        bm = block_mask if block_mask is not None else self._block_mask(fm, B, N, N, tok.device)
        out = self.attn.full_attn(h, cos, sin, block_mask=bm)
        tok = tok + out
        o = tok.view(B, T, self.tpf, C)
        x_img_o = rearrange(o[:, :, :H * W], "b t (h w) c -> b c t h w", h=H, w=W)
        x_side_o = rearrange(o[:, :, H * W:], "b t s c -> b c s t")
        return x_img_o, x_side_o
