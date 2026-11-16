"""Parameter split used by ``configure_optimizers`` (AdamW, ``wd_matrices_only=True``)."""
from __future__ import annotations


def split_muon_adamw_params(named_params):
    """Split ``(name, param)`` pairs into ``(matrices, others)``.

    ``matrices`` = every trainable 2-D weight or singleton-kernel tensor (trailing dims all 1) whose
    name is not an input embedder / output head / AdaLN modulation head; these get weight decay.
    Everything else (norms, biases, embedders, heads) goes to ``others`` with weight decay 0."""
    muon, adamw = [], []
    # name substrings that force the no-decay group even for a 2-D / foldable weight
    ADAMW_KEYS = ("action_emd", "pos_embedding", "rotary", "patch_embed", "unpatch",
                  "state_in", "state_out", "emb_layers", "cond_emb_layers")
    for name, p in named_params:
        if not p.requires_grad:
            continue
        lname = name.lower()
        foldable = (p.ndim == 2) or (p.ndim > 2 and p.shape[2:].numel() == 1)  # 2D or (o,i,1,1,1)
        if foldable and not any(k in lname for k in ADAMW_KEYS):
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw
