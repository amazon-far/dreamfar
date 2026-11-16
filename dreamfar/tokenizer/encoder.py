"""Tokenizer encoder: SSDD's native f16 conv encoder (``ssdd.models.vq_encoder.VQEncoder`` from the
facebook/SSDD package, an external dependency, see THIRD_PARTY.md) with the posterior mean projected onto
the per-pixel unit sphere. Parameter names are SSDD's, so the pretrained ``F16C16_M_256.safetensors``
``encoder.*`` tensors load strictly (126/126)."""
from __future__ import annotations

import torch
from safetensors.torch import load_file as safe_load_file
from ssdd.models.vq_encoder import VQEncoder


class SsddEncoder(VQEncoder):
    """(B,3,H,W) in [-1,1] -> (B,z_dim,H/patch_size,W/patch_size), unit L2 norm over channels."""

    def __init__(self, z_dim: int = 16, patch_size: int = 16, base_dim: int = 128, num_res_blocks: int = 2,
                 in_dim: int = 3, dropout: float = 0.0):
        # SSDD's standard config (double_z: a mean|logvar head); only the mean is used.
        super().__init__(**VQEncoder.get_config(z_dim, patch_size, base_dim=base_dim,
                                                num_res_blocks=num_res_blocks, in_dim=in_dim, dropout=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = super().forward(x).mean                 # posterior mean; logvar discarded
        return mean / (torch.norm(mean, dim=1, keepdim=True) + 1e-8)

    def load_pretrained(self, checkpoint: str) -> dict:
        """Strictly load ALL ``encoder.*`` tensors of an SSDD .safetensors (head included)."""
        sd = safe_load_file(checkpoint)
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        own = self.state_dict()
        missing = sorted(set(own) - set(enc))
        unexpected = sorted(set(enc) - set(own))
        mismatched = sorted(k for k in set(own) & set(enc) if tuple(own[k].shape) != tuple(enc[k].shape))
        if missing or unexpected or mismatched:
            raise RuntimeError(
                f"SSDD encoder load mismatch: missing={missing[:6]} ({len(missing)}), "
                f"unexpected={unexpected[:6]} ({len(unexpected)}), "
                f"shape-mismatched={mismatched[:6]} ({len(mismatched)})"
            )
        self.load_state_dict({**own, **enc}, strict=True)
        return {"loaded": len(enc), "own_keys": len(own)}
