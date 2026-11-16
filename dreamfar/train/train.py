"""Hydra training entrypoint: ``python -m dreamfar.train.train algorithm=dynamics name=<run> ...``."""
from __future__ import annotations

import hashlib
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dreamfar.train.experiment import Experiment

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _build_logger(cfg: DictConfig, output_dir: Path):
    if cfg.wandb.mode == "disabled":
        return None
    from lightning.pytorch.loggers.wandb import WandbLogger
    # A run id derived from the run name: every relaunch of the same name continues the same curve.
    return WandbLogger(name=str(cfg.name), id="id" + hashlib.sha256(str(cfg.name).encode()).hexdigest()[:14],
                       resume="allow", save_dir=str(output_dir), offline=cfg.wandb.mode != "online",
                       entity=cfg.wandb.entity, project=cfg.wandb.project, log_model=False,
                       config=OmegaConf.to_container(cfg))


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="config")
def run(cfg: DictConfig) -> None:
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print("outputs ->", output_dir, flush=True)
    Experiment(cfg, _build_logger(cfg, output_dir)).train()


if __name__ == "__main__":
    run()
