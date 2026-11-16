"""Tokenizer decoder: SSDD's U-ViT flow-matching decoder (``ssdd.models.ssdd.uvit.UViTDecoder`` from the
facebook/SSDD package, an external dependency, see THIRD_PARTY.md) with two additions: a trainable
latent-scale scalar ``z_scale`` and optional qk-norm (``attn_norm``) in the mid-block attention."""
from __future__ import annotations

import inspect
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file
from ssdd.models.ssdd.uvit import UViTDecoder as _SsddUViTDecoder
from ssdd.models.ssdd.uvit import UViTMiddleTransformer


class UViTDecoder(_SsddUViTDecoder):
    def __init__(self, attn_norm: Optional[str] = None, z_scale: Optional[float] = None, **kwargs):
        """``kwargs`` are SSDD's ``UViTDecoder`` arguments. ``attn_norm`` (e.g. ``"rms_norm"``) adds qk-norm
        to every mid-block attention. ``z_scale``: None = no parameter (forward unchanged); a float creates ONE
        trainable scalar applied to z at the top of forward (z = z * z_scale), not folded into weights."""
        super().__init__(**kwargs)
        if attn_norm is not None:
            # SSDD builds the mid block without qk-norm: rebuild it with the same arguments plus qk-norm,
            # then run SSDD's init again (it only initializes the new, not-yet-initialized modules).
            a = inspect.signature(_SsddUViTDecoder.__init__).bind(self, **kwargs)
            a.apply_defaults()
            a = a.arguments
            assert not a["learned_pos_embed"], "attn_norm with learned_pos_embed is not supported"
            dropout = a["dropout"] if isinstance(a["dropout"], (int, float)) else a["dropout"][-1]
            self.mid_block = UViTMiddleTransformer(
                inner_dim=self.mid_dim,
                dropout=dropout,
                num_layers=a["mid_nlayers"],
                norm_num_groups=a["norm_num_groups"],
                num_attention_heads=a["num_attention_heads"],
                rope_theta=a["mid_theta"],
                attn_window=a["attn_window"],
                eps=a["eps"],
                ada_norm=a["ada_norm"],
                ada_emb_dim=self.ch_level[0],
                learned_pos_embed=False,
                sample_size=None,
                relative_pos_embed=a["relative_pos_embed"],
                act_fn=a["vit_act_fn"],
                attn_norm=attn_norm,
            )
            self.init_weights(**(a["init"] or {}))
        self.z_scale = None if z_scale is None else nn.Parameter(torch.tensor(float(z_scale)))

    # Keys SSDD's checkpoint has no counterpart for: the qk-norm gains (left at `ones`) and z_scale.
    QK_NORM_SUFFIXES = ("attn1.norm_q.weight", "attn1.norm_k.weight")
    OWN_ONLY_KEYS = ("z_scale",)

    @torch.no_grad()
    def load_pretrained(self, checkpoint: str, prefix: str = "decoder.") -> dict:
        """Load SSDD's ``decoder.*`` tensors; every key is accounted for (missing must be qk-norm/z_scale)."""
        sd = safe_load_file(checkpoint)
        dec = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not dec:
            raise RuntimeError(f"no '{prefix}*' tensors in {checkpoint}")

        own = self.state_dict()
        unexpected = sorted(set(dec) - set(own))
        mismatched = sorted(k for k in set(own) & set(dec) if tuple(own[k].shape) != tuple(dec[k].shape))
        missing = sorted(set(own) - set(dec))
        qk = [k for k in missing if k.endswith(self.QK_NORM_SUFFIXES)]
        own_only = [k for k in missing if k in self.OWN_ONLY_KEYS]
        unexplained = sorted(set(missing) - set(qk) - set(own_only))
        if unexpected or mismatched or unexplained:
            raise RuntimeError(
                f"SSDD decoder load mismatch ({checkpoint}):\n"
                f"  shape-mismatched ({len(mismatched)}): {mismatched[:8]}\n"
                f"  unexpected ({len(unexpected)}): {unexpected[:8]}\n"
                f"  missing-and-unaccounted ({len(unexplained)}): {unexplained[:8]}"
            )
        self.load_state_dict({**own, **dec}, strict=True)
        return {"ckpt_tensors": len(dec), "own_keys": len(own), "loaded": len(dec),
                "missing_qk_norm": qk, "missing_own_only": own_only,
                "z_scale": None if self.z_scale is None else float(self.z_scale)}

    def forward(self, x, t, z):
        # t=0.0 -> no noise ; t=1.0 -> full noise (t is already multiplied by the trainer's timescale)
        if self.z_scale is not None:
            z = z * self.z_scale
        return super().forward(x, t, z)
