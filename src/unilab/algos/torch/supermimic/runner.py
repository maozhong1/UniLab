"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy.

The env is a UniLab ``NpEnv`` exposing the ``sm_*`` data interface of
``unilab.envs.motion_tracking.h2.H2SuperMimicEnv``; the replay buffer, world model, policy and
critic live on the learner device.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .heads import AnalyticHeads, TrackingRewardCfg
from .kinematics import HumanoidKinematics
from .motion_library import MotionLibrary
from .nets import Critic, CriticCfg, Policy, PolicyCfg
from .replay_buffer import ReplayBuffer
from .trainers import PolicyTrainer, PolicyTrainerCfg, WMTrainerCfg, WorldModelTrainer
from .world_model import WorldModel, WorldModelCfg, set_mlp_autocast


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps
    updates_per_env_step: float = 0.4  # update-to-data ratio
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50
    checkpoint_interval: int = 4000
    max_hours: float | None = None
    # Optional LR restart for fine-tuning a resumed run: from iteration ``lr_restart_iteration`` the
    # LR ramps linearly from ``lr_final_ratio`` to ``lr_restart_peak_ratio`` over
    # ``lr_restart_warmup_iterations`` (counted from the first update after the restart), then
    # decays by cosine back to ``lr_final_ratio`` over ``lr_decay_fraction`` of the remaining run.
    lr_restart_iteration: int | None = None
    lr_restart_peak_ratio: float = 0.2
    lr_restart_warmup_iterations: int = 2000


@dataclass
class SuperMimicCfg:
    seed: int = 0
    replay_capacity: int = 1000
    library_fp16: bool = False
    # MLP matmul precision: "float32" or "bfloat16" (autocast; FK, integration, rewards stay fp32).
    mlp_dtype: str = "float32"
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)


def cosine_lr(
    step: int, total_steps: int, base_lr: float, final_ratio: float, decay_fraction: float
) -> float:
    """Cosine decay to ``final_ratio`` over ``decay_fraction`` of training, then constant."""
    decay_steps = max(1, int(total_steps * decay_fraction))
    if step >= decay_steps:
        return base_lr * final_ratio
    return base_lr * (
        final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + math.cos(math.pi * step / decay_steps))
    )


def scheduled_lr(step: int, sch: ScheduleCfg, base_lr: float) -> float:
    """LR at ``step``: the base cosine schedule, or the restart schedule once it has begun."""
    start = sch.lr_restart_iteration
    if start is None or step < start:
        return cosine_lr(
            step, sch.total_iterations, base_lr, sch.lr_final_ratio, sch.lr_decay_fraction
        )
    lo, hi = sch.lr_final_ratio, sch.lr_restart_peak_ratio
    k = max(0, step - start - sch.warmup_iterations)
    warm = max(1, sch.lr_restart_warmup_iterations)
    if k < warm:
        return base_lr * (lo + (hi - lo) * k / warm)
    remaining = sch.total_iterations - start - sch.warmup_iterations - warm
    decay_steps = max(1, int(remaining * sch.lr_decay_fraction))
    x = min(k - warm, decay_steps) / decay_steps
    return base_lr * (lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * x)))


def build_models(
    env: Any, cfg: SuperMimicCfg, device: str
) -> tuple[WorldModel, Policy, Critic, MotionLibrary]:
    """World model (with analytic heads), policy, critic and the device-side reference library for ``env``."""
    ecfg = env.cfg
    nj = env.num_joints
    cfg.policy.num_joints = nj
    cfg.policy.history = ecfg.history_length
    cfg.policy.future_frames = ecfg.future_frames
    cfg.world_model.state_spec.num_joints = nj
    cfg.world_model.state_spec.dt = ecfg.ctrl_dt
    cfg.world_model.height_threshold = ecfg.root_height_threshold
    cfg.world_model.tilt_threshold = ecfg.root_tilt_threshold
    fk = HumanoidKinematics(env._backend.model, device=device)
    q0 = torch.as_tensor(env.default_joint_pos, device=device)
    reward_cfg = TrackingRewardCfg(**asdict(ecfg.reward), dt=ecfg.ctrl_dt)
    heads = AnalyticHeads(
        fk,
        ecfg.tracked_bodies,
        ecfg.anchor_body_name,
        ecfg.root_body_name,
        torch.as_tensor(env.soft_joint_limits, device=device),
        q0,
        reward_cfg,
    )
    wm = WorldModel(cfg.world_model, heads, device)
    policy = Policy(cfg.policy).to(device)
    critic = Critic(cfg.critic, cfg.policy).to(device)
    if cfg.mlp_dtype not in ("float32", "bfloat16"):
        raise ValueError(f"mlp_dtype must be float32 or bfloat16, got {cfg.mlp_dtype!r}")
    for m in (wm, policy, critic):
        set_mlp_autocast(m, torch.bfloat16 if cfg.mlp_dtype == "bfloat16" else None)
    lib = MotionLibrary.from_env_library(
        env.library,
        fk,
        ecfg.tracked_bodies,
        device,
        torch.float16 if cfg.library_fp16 else torch.float32,
    )
    return wm, policy, critic, lib


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


class SuperMimicRunner:
    def __init__(
        self,
        env: Any,
        cfg: SuperMimicCfg,
        device: str,
        log_dir: str | Path,
        resume: str | Path | None = None,
    ):
        self.env, self.cfg = env, cfg
        self.device = torch.device(device)
        torch.manual_seed(cfg.seed)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.wm, self.policy, self.critic, self.lib = build_models(env, cfg, device)
        (self.log_dir / "supermimic_config.json").write_text(
            json.dumps({"algo": asdict(cfg), "env": asdict(env.cfg)}, indent=2, default=str)
        )
        if self.device.type == "xpu" and (cfg.wm_trainer.compile or cfg.policy_trainer.compile):
            # Horizontal fusion of independent small kernels: the compiled FK / reward graphs are
            # launch-bound on XPU (~25 us per Triton launch), and this cuts the launch count ~3x.
            torch._inductor.config.combo_kernels = True
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(
            self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer, env.cfg.obs_noise_half
        )
        self.buffer = ReplayBuffer(
            env.num_envs,
            cfg.replay_capacity,
            env.state_dim,
            env.num_joints,
            env.xi_dim,
            self.device,
        )
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        print(
            f"[SuperMimic] params: world model {n_params(self.wm) / 1e6:.2f}M, policy {n_params(self.policy) / 1e6:.2f}M, critic {n_params(self.critic) / 1e6:.3f}M"
        )
        print(f"[SuperMimic] {env.num_envs} envs, device {self.device}, log dir {self.log_dir}")
        self.max_step_reward = self.wm.heads.reward_cfg.max_step_reward
        self.iteration = 0
        self.start_iteration = 0
        self.num_updates = 0
        self._utd_acc = 0.0
        if resume is not None:
            self.load(resume)

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        torch.save(
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                "bin_return": self.env.sampler.bin_return,
                "bin_memory": self.env.sampler.bin_memory,
                "clip_names": self.env.library.clip_names,
                "config": asdict(self.cfg),
            },
            self.log_dir / name,
        )

    def load(self, path: str | Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
        self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
        self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
        if tuple(ck.get("clip_names", ())) == self.env.library.clip_names:
            self.env.sampler.bin_return[:] = ck["bin_return"]
            self.env.sampler.bin_memory[:] = ck["bin_memory"]
        else:
            print(
                "[SuperMimic] clip set differs from the checkpoint: sampler statistics start afresh"
            )
        self.iteration = self.start_iteration = int(ck["iteration"])
        self.num_updates = int(ck["num_updates"])
        print(f"[SuperMimic] resumed from {path} at iteration {self.iteration}")

    # ------------------------------------------------------------------ helpers

    def _t(self, x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(x), device=self.device).to(dtype)

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray]) -> torch.Tensor:
        return self.policy(self._t(obs["obs"]), self._t(obs["goal"]))

    # ------------------------------------------------------------------ main loop

    def run(self) -> None:
        cfg, sch, env = self.cfg, self.cfg.schedule, self.env
        n = env.num_envs
        state = env.init_state()
        all_ids = torch.arange(n, device=self.device)
        prev_s = self._t(env.sm_state())
        prev_xi = self._t(env.sm_xi())
        prev_frame = self._t(env.frame, torch.long)
        ep_start = torch.ones(n, dtype=torch.bool, device=self.device)
        ep_reward = np.zeros(n)
        ep_len = np.zeros(n)
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        finished_returns: list[float] = []
        finished_lens: list[float] = []
        t_start = last_log = time.time()

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect one vectorised env step and stream it into the replay buffer
            t0 = time.perf_counter()
            action = self.act(state.obs)
            state = env.step(action.cpu().numpy())
            a_exec = self._t(env.sm_executed_action)
            term = state.terminated.copy()
            done = term | state.truncated
            reward = self._t(state.reward)
            self.buffer.add(
                all_ids,
                s=prev_s,
                a=action,
                a_exec=a_exec,
                xi=prev_xi,
                r=reward,
                done=self._t(term),
                frame=prev_frame,
                clip=self.lib.frame_clip[prev_frame],
                terminal=torch.zeros(n, dtype=torch.bool, device=self.device),
                ep_start=ep_start,
            )
            ep_reward += state.reward
            ep_len += 1
            if done.any():
                ids_np = np.flatnonzero(done)
                ids = self._t(ids_np, torch.long)
                # Terminal row: the state reached by the last action (no transition out of it).
                m = len(ids_np)
                zeros_a = torch.zeros(m, env.num_joints, device=self.device)
                term_frame = self._t(env.sm_frame_pre_reset[ids_np], torch.long)
                self.buffer.add(
                    ids,
                    s=self._t(env.sm_state_pre_reset[ids_np]),
                    a=zeros_a,
                    a_exec=zeros_a,
                    xi=prev_xi[ids],
                    r=torch.zeros(m, device=self.device),
                    done=self._t(term[ids_np]),
                    frame=term_frame,
                    clip=self.lib.frame_clip[term_frame],
                    terminal=torch.ones(m, dtype=torch.bool, device=self.device),
                    ep_start=torch.zeros(m, dtype=torch.bool, device=self.device),
                )
                self.buffer.end_episode(ids)
                finished_returns.extend(
                    (
                        ep_reward[ids_np] / (self.max_step_reward * np.maximum(ep_len[ids_np], 1))
                    ).tolist()
                )
                finished_lens.extend(ep_len[ids_np].tolist())
                ep_reward[ids_np] = 0.0
                ep_len[ids_np] = 0.0
            prev_s = self._t(env.sm_state())
            prev_xi = self._t(env.sm_xi())
            prev_frame = self._t(env.frame, torch.long)
            ep_start = self._t(done, torch.bool)
            timers["sim"] += time.perf_counter() - t0
            self.iteration += 1

            # ---------------- (2)+(3) updates at the configured update-to-data ratio
            if self.iteration - self.start_iteration >= sch.warmup_iterations:
                self._utd_acc += sch.updates_per_env_step
                while self._utd_acc >= 1.0:
                    self._utd_acc -= 1.0
                    lr = scheduled_lr(self.iteration, sch, cfg.wm_trainer.lr)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
                    t0 = time.perf_counter()
                    m_wm = self.wm_trainer.update(self.buffer)
                    _sync(self.device)
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    for _ in range(sch.policy_updates_per_wm_update):
                        m_pi = self.policy_trainer.update(self.buffer)
                    _sync(self.device)
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in m_wm.items():
                        agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
                    for k, v in m_pi.items():
                        agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
                    agg["lr"] = lr
                    agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                self._log(
                    agg,
                    agg_n,
                    finished_returns,
                    finished_lens,
                    timers,
                    t_start,
                    last_log,
                    state.info.get("log"),
                )
                last_log = time.time()
                agg, agg_n = {}, 0
                finished_returns, finished_lens = [], []
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if sch.max_hours is not None and (time.time() - t_start) / 3600 > sch.max_hours:
                print("[SuperMimic] max_hours reached, stopping")
                break
        self.save("ckpt_latest.pt")
        self.writer.flush()

    def _log(
        self, agg, agg_n, finished_returns, finished_lens, timers, t_start, last_log, env_log
    ) -> None:
        it, sch, w = self.iteration, self.cfg.schedule, self.writer
        elapsed = time.time() - t_start
        if agg_n > 0:
            for k, v in agg.items():
                w.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
        tr = float(np.mean(finished_returns)) if finished_returns else float("nan")
        if finished_returns:
            w.add_scalar("train/episode_norm_return", tr, it)
            w.add_scalar("train/episode_len", float(np.mean(finished_lens)), it)
        err = None
        if self.buffer.size > 2048:
            err = self.wm_trainer.online_error(self.buffer, n=1024, k=self.cfg.wm_trainer.window)
            w.add_scalar("wm/online_nse_1step", float(err[0]), it)
            w.add_scalar("wm/online_nse_kstep", float(err[-1]), it)
        for k, v in timers.items():
            w.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
        w.add_scalar("time/hours", elapsed / 3600, it)
        w.add_scalar("time/updates", self.num_updates, it)
        w.add_scalar("sampler/entropy", self.env.sampler.entropy, it)
        w.add_scalar("sampler/bin_return_mean", float(self.env.sampler.bin_return.mean()), it)
        for k, v in (env_log or {}).items():
            w.add_scalar(k, v, it)
        den = max(agg_n, 1)
        print(
            f"it {it} | {elapsed / 3600:.2f} h | {sch.log_interval / max(time.time() - last_log, 1e-9):.2f} it/s | updates {self.num_updates} | "
            f"train_ret {tr:.3f} (n={len(finished_returns)}) | "
            f"wm nse1 {agg.get('wm/nse_1step', float('nan')) / den:.4f} nse{self.cfg.wm_trainer.window} {(float(err[-1]) if err is not None else float('nan')):.4f} | "
            f"imag_ret {agg.get('policy/imag_return_undisc', float('nan')) / den:.3f} surv {agg.get('policy/imag_survival_H', float('nan')) / den:.3f} | "
            f"lr {agg.get('lr', float('nan')):.2e} | "
            f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s",
            flush=True,
        )
