"""Auxiliary tokenizer losses: VGG-LPIPS + REPA against a frozen LingBot-Vision teacher (the loss terms
used by SSDD's tokenizer training recipe). Both frozen nets are held in plain lists, OUTSIDE the module
tree, so they never enter the state_dict or DDP sync."""
from __future__ import annotations

import sys
from typing import Optional

import torch
import torch.nn as nn

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    return module


class LingbotEncoder(nn.Module):
    """Frozen LingBot-Vision ViT/16 patch-feature extractor; grid is pinned to ``res / 16`` per side.

    Needs the external ``lingbot_vision`` package (github.com/robbyant/lingbot-vision, Apache-2.0):
    either pip-installed or a clone passed as ``repo_dir``; ``weights_dir`` holds the variant's weights."""

    def __init__(self, variant: str = "small", res: int = 512, weights_dir: Optional[str] = None,
                 repo_dir: Optional[str] = None):
        super().__init__()
        from torchvision.transforms import Normalize

        self.res = int(res)
        self.variant = str(variant)
        self._normalize = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        if repo_dir and repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from lingbot_vision import load_pretrained_backbone

        bb, embed = load_pretrained_backbone(weights_dir, variant=variant, device="cpu", dtype=torch.bfloat16)
        bb.eval()
        for p in bb.parameters():
            p.requires_grad = False
        self.model = bb
        self.out_dim = int(embed)
        self.base_patch_size = int(bb.patch_size)
        self.fixed_grid = g = self.res // self.base_patch_size
        print(f"[tok-repa] teacher=lingbot/{variant} embed={embed} patch={bb.patch_size} "
              f"res={res} -> {g}x{g}={g * g} tokens", flush=True)
        freeze(self)

    def forward(self, x, target_n_tokens=None):
        if self.training:
            self.eval()
        # x is the tokenizer's [-1,1] image; LingBot wants ImageNet-normalized [0,1] at self.res.
        x = self._normalize(((x + 1) / 2).clamp(0, 1))
        x = torch.nn.functional.interpolate(x, (self.res, self.res), mode="bicubic", align_corners=False)
        out = self.model(x.to(torch.bfloat16), is_training=True)
        tok = out["x_norm_patchtokens"] if isinstance(out, dict) else out
        if target_n_tokens is not None and tok.shape[1] != target_n_tokens:
            g_dst = int(round(target_n_tokens ** 0.5))
            raise RuntimeError(
                f"REPA teacher/student grid mismatch: teacher gives {tok.shape[1]} tokens "
                f"(res={self.res} / patch {self.base_patch_size} = {self.fixed_grid} per side) "
                f"but the decoder mid-block has {target_n_tokens} ({g_dst} per side). Set "
                f"repa.lingbot_res={g_dst * self.base_patch_size}."
            )
        return tok.float()


class REPALoss(nn.Module):
    """REPA (Yu et al., 2024, arXiv 2410.06940): align an intermediate decoder representation with frozen
    teacher patch features. A forward hook keeps the (B, N, C) output of
    ``decoder.mid_block.transformer_blocks[i_extract]`` during training; an MLP projector (``n_layers``
    Linear layers with SiLU in between) maps it to the teacher width, and the loss is one minus the mean
    token-wise cosine similarity, computed in fp32."""

    def __init__(self, decoder: nn.Module, n_layers: int = 2, i_extract: int = 4,
                 lingbot_variant: str = "small", lingbot_res: int = 512,
                 lingbot_weights_dir: Optional[str] = None, lingbot_repo_dir: Optional[str] = None,
                 image_size: Optional[int] = None):
        super().__init__()
        teacher = LingbotEncoder(variant=lingbot_variant, res=lingbot_res, weights_dir=lingbot_weights_dir,
                                 repo_dir=lingbot_repo_dir)
        # Student grid = image_size / 2**(levels-1) must equal the teacher grid (no silent resample).
        s_grid = None if image_size is None else int(image_size) // 2 ** (len(decoder.ch_level) - 1)
        t_grid = teacher.fixed_grid
        if s_grid is not None and s_grid != t_grid:
            raise ValueError(
                f"REPA grid mismatch: decoder mid-block is {s_grid}x{s_grid} tokens (image_size="
                f"{image_size} / 2**{len(decoder.ch_level) - 1}) but the teacher gives {t_grid}x{t_grid} "
                f"(lingbot_res={lingbot_res} / patch {teacher.base_patch_size}). Set "
                f"repa.lingbot_res={s_grid * teacher.base_patch_size}."
            )
        self._extractor = [teacher]                    # outside the module tree: no state_dict / DDP
        self.register_buffer("_dev_probe", torch.zeros(1), persistent=False)

        widths = [decoder.mid_dim] * n_layers + [teacher.out_dim]
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
        self.repa_mlp = nn.Sequential(*layers)

        blocks = decoder.mid_block.transformer_blocks
        i_extract = min(i_extract, len(blocks) // 2)
        print(f"[tok-repa] hook on decoder.mid_block.transformer_blocks[{i_extract}] of {len(blocks)}; "
              f"source_dim={decoder.mid_dim} -> target_dim={teacher.out_dim}; "
              f"student_grid={s_grid} teacher_grid={t_grid}", flush=True)
        blocks[i_extract].register_forward_hook(self._keep_features)
        self._features = None

    @property
    def features_extractor(self) -> nn.Module:
        return self._extractor[0]

    def _apply(self, fn, *args, **kwargs):
        """Carry the hidden teacher along on .to()/.cuda()/Lightning's device moves."""
        out = super()._apply(fn, *args, **kwargs)
        self._extractor[0] = self._extractor[0].to(device=self._dev_probe.device)
        return out

    def _keep_features(self, module, inp, out):
        if self.training:
            self._features = out

    def forward(self, x_gt):
        student, self._features = self.repa_mlp(self._features), None
        with torch.no_grad():
            target = self.features_extractor(x_gt, target_n_tokens=student.shape[1])
        assert student.shape == target.shape, f"REPA shape mismatch {student.shape} != {target.shape}"
        with torch.autocast("cuda", enabled=False):
            cos = torch.nn.functional.cosine_similarity(student.float(), target.float(), dim=2, eps=1e-5)
            return 1 - cos.mean()


class AuxLosses(nn.Module):
    """LPIPS(vgg) + optional REPA. ``repa`` is the ``algorithm.repa`` block (``warmup_steps`` is
    consumed by the LightningModule and dropped here); ``image_size`` is the tokenizer input size."""

    def __init__(self, decoder: nn.Module, lpips: bool = True, repa: Optional[dict] = None,
                 image_size: Optional[int] = None):
        super().__init__()
        self.register_buffer("_dev_probe", torch.zeros(1), persistent=False)
        self._lpips = []
        if lpips:
            import lpips as lpips_lib
            self._lpips = [freeze(lpips_lib.LPIPS(net="vgg"))]
        self.repa_loss = None
        if repa is not None:
            kw = {k: v for k, v in repa.items() if k != "warmup_steps"}
            self.repa_loss = REPALoss(decoder, image_size=image_size, **kw)

    @property
    def lpips_loss(self):
        return self._lpips[0] if self._lpips else None

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        if self._lpips:
            self._lpips[0] = self._lpips[0].to(device=self._dev_probe.device)
        return out

    def forward(self, x_gt: torch.Tensor, x0_pred: torch.Tensor) -> dict:
        losses = {}
        if self.repa_loss is not None:
            losses["repa"] = self.repa_loss(x_gt)
        if self.lpips_loss is not None:
            losses["lpips"] = self.lpips_loss(x_gt, x0_pred).mean()
        return losses
