"""Print the fully resolved config for a set of Hydra overrides (the model-zoo / inference config format).

    python -m dreamfar.train.dump_config name=x algorithm=dynamics dataset=latents dataset.latent_root=... > cfg.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")

if __name__ == "__main__":
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        cfg.pop("hydra", None)
    print(OmegaConf.to_yaml(cfg), end="")
