"""World-model fitting on replay windows (Eq. 1) and short-horizon actor-critic through the frozen
world model (SuperMimic §3.3, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .imagination import ImagStep, gae_targets, imagine, obs_noise_half, policy_objective
from .motion_library import MotionLibrary
from .nets import Critic, Policy
from .replay_buffer import ReplayBuffer
from .world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = None  # None | "default" | torch.compile mode


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg):
        self.wm, self.lib, self.cfg = wm, lib, cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        for _ in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            r = self.lib.ref(
                w.frame.reshape(-1)
            )  # only the reference roots are needed (termination features)
            k, n = w.frame.shape
            loss, m = self.wm.window_loss(
                w.states,
                w.a_exec,
                w.xi,
                w.done,
                r.root_pos_w.reshape(k, n, 3),
                r.root_quat_w.reshape(k, n, 4),
            )
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip)
            self.opt.step()
            m["loss"] = loss.detach()
            m["grad_norm"] = gn.detach()
            for key, v in m.items():
                agg[key] = agg.get(key, 0.0) + v
        return {key: float(v) / self.cfg.batches_per_update for key, v in agg.items()}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = None


class PolicyTrainer:
    def __init__(
        self,
        wm: WorldModel,
        policy: Policy,
        critic: Critic,
        lib: MotionLibrary,
        cfg: PolicyTrainerCfg,
        obs_noise: tuple[float, ...],
    ):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        nj = lib.default_joint_pos.numel()
        self.noise_half = obs_noise_half(nj, obs_noise).to(lib.device) if cfg.obs_noise else None
        self.step_fn = ImagStep(
            wm,
            lib.default_joint_pos,
            compile_mode=cfg.compile,
            policy=policy,
            noise_half=self.noise_half,
        )

    def set_lr(self, lr: float) -> None:
        # The critic follows the policy schedule (as in the reference implementation).
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = lr

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size, self.policy.cfg.history)
        xi_norm = self.wm.spec.normalize_xi(start.xi)

        # ---- actor: BPTT through the frozen world model and critic
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        ro = imagine(
            self.wm,
            self.policy,
            self.lib,
            start,
            cfg.horizon,
            self.policy.cfg.future_frames,
            self.step_fn,
            self.noise_half,
        )
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        v_H = (
            self.critic(feat[-1], ro.goals[-1], xi_norm, ro.e_root[-1]) if cfg.use_critic else None
        )
        J = policy_objective(ro, v_H, cfg.gamma)
        loss = -J / cfg.horizon
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        self.opt.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)

        metrics = {
            "imag_objective": float(J.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(gn),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
        }
        metrics.update({f"imag_r/{k}": v for k, v in ro.reward_terms.items()})

        # ---- critic: fit to the (fixed) TD(lambda) targets of this rollout
        if cfg.use_critic:
            with torch.no_grad():
                feat_d, goals_d, e_root_d = feat.detach(), ro.goals.detach(), ro.e_root.detach()
                xi_rep = xi_norm[None].expand(H1, n, -1)
                values = self.critic(
                    feat_d.reshape(H1 * n, -1),
                    goals_d.reshape(H1 * n, -1),
                    xi_rep.reshape(H1 * n, -1),
                    e_root_d.reshape(H1 * n, -1),
                ).reshape(H1, n)
                targets = gae_targets(
                    ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda
                )
                X = (
                    feat_d[:-1].reshape(-1, feat_d.shape[-1]),
                    goals_d[:-1].reshape(-1, goals_d.shape[-1]),
                    xi_rep[:-1].reshape(-1, xi_rep.shape[-1]),
                    e_root_d[:-1].reshape(-1, 4),
                )
                Y = targets.reshape(-1)
            N = Y.shape[0]
            mb = N // cfg.critic_minibatches
            last = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    closs = torch.nn.functional.mse_loss(self.critic(*(x[idx] for x in X)), Y[idx])
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = float(last)
            metrics["critic_value_mean"] = float(values.mean())
            metrics["critic_target_mean"] = float(targets.mean())
        return metrics
