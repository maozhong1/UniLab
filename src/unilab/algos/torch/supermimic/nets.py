"""Policy (goal encoder + shared decoder) and critic (SuperMimic Table A.4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .heads import obs_term_dims
from .world_model import MLP


@dataclass
class PolicyCfg:
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = 32
    history: int = 10
    future_frames: int = 10
    # Fixed per-term observation scales (divide): joint_pos, joint_vel, base_ang_vel, projected_gravity, actions.
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0)
    # Fixed goal feature scales: joint_pos N, joint_vel N, grav 3, ang_vel_b 3, lin_vel_h 3, height 1.
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    decoder_last_gain: float = 0.01
    num_joints: int = 29  # set from the env by the runner

    @property
    def obs_dim(self) -> int:
        return self.history * (3 * self.num_joints + 6)

    @property
    def goal_dim(self) -> int:
        return self.future_frames * (2 * self.num_joints + 10)


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    parts = [
        torch.full((cfg.history * d,), 1.0 / s)
        for d, s in zip(obs_term_dims(cfg.num_joints), cfg.obs_term_scale)
    ]
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    nj = cfg.num_joints
    per_frame = torch.cat(
        [torch.full((n,), 1.0 / s) for n, s in zip((nj, nj, 3, 3, 3, 1), cfg.goal_term_scale)]
    )
    return per_frame.repeat(cfg.future_frames)


class MotionEncoder(nn.Module):
    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.mlp = MLP(cfg.goal_dim, cfg.encoder_hidden, cfg.latent_dim)
        self.register_buffer("scale", _goal_scale_vector(cfg))

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(goal * self.scale))


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.mlp = MLP(
            cfg.latent_dim + cfg.obs_dim,
            cfg.decoder_hidden,
            cfg.num_joints,
            last_gain=cfg.decoder_last_gain,
        )
        self.register_buffer("scale", _obs_scale_vector(cfg))

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z, obs_hist * self.scale], dim=-1))


class Policy(nn.Module):
    """a = dec(enc_motion(goal), obs_hist); deterministic, input scaling inside the network."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        self.decoder = Decoder(cfg)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoders["motion"](goal), obs_hist)


class Critic(nn.Module):
    """V(canonical state 13+2N, goal, normalised xi N+4, root error 4) -> scalar."""

    def __init__(self, cfg: CriticCfg, pcfg: PolicyCfg):
        super().__init__()
        nj = pcfg.num_joints
        self.mlp = MLP((13 + 2 * nj) + pcfg.goal_dim + (nj + 4) + 4, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg))

    def forward(
        self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, e_root: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, e_root], dim=-1)).squeeze(
            -1
        )
