"""SuperMimic training entrypoint (assembly only): Hydra -> registry env -> SuperMimicRunner.

Usage:
    uv run scripts/train_supermimic.py                                    # H2, full data, CUDA/XPU/CPU auto
    uv run scripts/train_supermimic.py task=h2_supermimic/mujoco_cpu_smoke
    uv run scripts/train_supermimic.py algo.num_envs=1024 training.device=cuda:0
    uv run scripts/train_supermimic.py algo.resume=logs/supermimic/H2SuperMimic/<run>/ckpt_latest.pt
"""

import datetime
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.algos.torch.supermimic import SuperMimicCfg, SuperMimicRunner
from unilab.base.registry import apply_cfg_overrides
from unilab.training import create_env, ensure_registries, get_log_root


def resolve_device(cfg: DictConfig) -> str:
    if cfg.training.device:
        return str(cfg.training.device)
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def algo_cfg(cfg: DictConfig) -> SuperMimicCfg:
    out = SuperMimicCfg()
    apply_cfg_overrides(
        out, cast(dict[str, Any], OmegaConf.to_container(cfg.algo.supermimic, resolve=True))
    )
    return out


@hydra.main(version_base="1.3", config_path="../conf/supermimic", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    device = resolve_device(cfg)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path(cfg.training.log_dir) if cfg.training.log_dir else get_log_root(ROOT_DIR, cfg)
    log_dir = log_root / str(cfg.training.task_name) / f"{timestamp}_{cfg.training.sim_backend}"
    env_override = (
        cast(dict[str, Any], OmegaConf.to_container(cfg.env, resolve=True)) if cfg.env else {}
    )
    env_override.setdefault("seed", int(cfg.algo.supermimic.seed))
    env = create_env(cfg, num_envs=int(cfg.algo.num_envs), env_cfg_override=env_override)
    runner = SuperMimicRunner(env, algo_cfg(cfg), device, log_dir, resume=cfg.algo.resume)
    runner.run()


if __name__ == "__main__":
    main()
