"""Imagined rollouts through the world model, policy objective (Eq. 2) and critic targets (Eq. 3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .heads import (
    REWARD_TERM_NAMES,
    ReferenceFrame,
    analytic_observation,
    flatten_history_term_major,
)
from .motion_library import MotionLibrary
from .nets import Policy
from .replay_buffer import StartBatch
from .world_model import WorldModel


def obs_noise_half(nj: int, half: tuple[float, ...] = (0.01, 0.5, 0.2, 0.05, 0.0)) -> torch.Tensor:
    """Uniform sensor-noise half-widths in the (3N+6)-d observation order (same as the env)."""
    return torch.cat([torch.full((d,), h) for d, h in zip((nj, nj, 3, 3, nj), half)])


class ImagStep:
    """One imagined step (dynamics, FK reward, termination logits, next clean observation) as one
    function so ``torch.compile`` can fuse the many small kernels of the per-body FK loop."""

    def __init__(
        self,
        wm: WorldModel,
        default_joint_pos: torch.Tensor,
        compile_mode: str | None = None,
        policy: Policy | None = None,
        noise_half: torch.Tensor | None = None,
    ):
        self.wm = wm
        self.q0 = default_joint_pos
        self.policy = policy
        self.noise_half = noise_half
        self._fn = self._step
        self._pfn = self._policy_step
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)
            # Fusing the policy forward, history update and sensor noise into the same graph removes
            # most of the per-step kernel launches, which bound the update on XPU.
            self._pfn = torch.compile(self._policy_step, mode=mode, dynamic=False)

    def _policy_step(
        self,
        hist,
        goal,
        s,
        a_prev,
        xi,
        ref_body_pos,
        ref_body_quat,
        ref_body_lin,
        ref_body_ang,
        ref_root_pos,
        ref_root_quat,
    ):
        """Policy -> world-model step -> noisy observation pushed into the (T, n, D) history."""
        a = self.policy(flatten_history_term_major(hist), goal)
        s_next, r, term_vec, logits, obs_clean = self._step(
            s,
            a,
            a_prev,
            xi,
            ref_body_pos,
            ref_body_quat,
            ref_body_lin,
            ref_body_ang,
            ref_root_pos,
            ref_root_quat,
        )
        if self.noise_half is not None:
            obs_clean = obs_clean + (torch.rand_like(obs_clean) * 2 - 1) * self.noise_half
        hist = torch.cat([hist[1:], obs_clean[None]], dim=0)
        return a, hist, s_next, r, term_vec, torch.sigmoid(logits)

    def policy_step(self, hist, goal, s, a_prev, xi, ref: ReferenceFrame):
        return self._pfn(
            hist,
            goal,
            s,
            a_prev,
            xi,
            ref.body_pos_w,
            ref.body_quat_w,
            ref.body_lin_vel_w,
            ref.body_ang_vel_w,
            ref.root_pos_w,
            ref.root_quat_w,
        )

    def _step(
        self,
        s,
        a,
        a_prev,
        xi,
        ref_body_pos,
        ref_body_quat,
        ref_body_lin,
        ref_body_ang,
        ref_root_pos,
        ref_root_quat,
    ):
        ref = ReferenceFrame(
            ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat
        )
        s_next = self.wm.step(s, a, xi)
        terms = self.wm.heads.reward(s_next, a, a_prev, ref)
        term_vec = torch.stack([terms[k] for k in REWARD_TERM_NAMES], dim=0)
        logits = self.wm.termination_logits(s_next, ref)
        obs_clean = analytic_observation(s_next, a, xi, self.q0)
        return s_next, terms["total"], term_vec, logits, obs_clean

    def __call__(self, s, a, a_prev, xi, ref: ReferenceFrame):
        return self._fn(
            s,
            a,
            a_prev,
            xi,
            ref.body_pos_w,
            ref.body_quat_w,
            ref.body_lin_vel_w,
            ref.body_ang_vel_w,
            ref.root_pos_w,
            ref.root_quat_w,
        )


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, S)
    actions: torch.Tensor  # (H, n, N)
    rewards: torch.Tensor  # (H, n) r_{t+1}
    dhat: torch.Tensor  # (H+1, n) predicted termination probability of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n) c_t
    goals: torch.Tensor  # (H+1, n, K*(2N+10))
    e_root: torch.Tensor  # (H+1, n, 4)
    reward_terms: dict


def imagine(
    wm: WorldModel,
    policy: Policy,
    lib: MotionLibrary,
    start: StartBatch,
    horizon: int,
    future_frames: int,
    step_fn: ImagStep,
    noise_half: torch.Tensor | None,
) -> ImagRollout:
    device = start.state.device
    n = start.state.shape[0]
    q0 = lib.default_joint_pos

    def add_noise(o):
        return o if noise_half is None else o + (torch.rand_like(o) * 2 - 1) * noise_half

    # Observation history from the logged states / actions (noise re-sampled per frame).
    hist = [
        add_noise(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, q0))
        for i in range(start.hist_states.shape[0])
    ]
    s, a_prev = start.state, start.prev_action
    # Target frames of the whole rollout are known up front (frame + t, clamped at the clip end).
    last = lib.last_frame_of(start.frame)
    frames_all = torch.minimum(
        start.frame[None, :] + torch.arange(horizon + 1, device=device)[:, None], last[None, :]
    )
    ref_all = lib.reference_at(frames_all.reshape(-1))

    def ref_at(t):
        sl = slice(t * n, (t + 1) * n)
        return ReferenceFrame(
            ref_all.body_pos_w[sl],
            ref_all.body_quat_w[sl],
            ref_all.body_lin_vel_w[sl],
            ref_all.body_ang_vel_w[sl],
            ref_all.root_pos_w[sl],
            ref_all.root_quat_w[sl],
        )

    # Goal windows and root-error features do not depend on the rollout, so they are computed for
    # all H+1 steps at once (a few big kernels instead of H+1 small ones).
    goals_all = lib.goal_window(frames_all.reshape(-1), future_frames).reshape(horizon + 1, n, -1)

    ref = ref_at(0)
    states, actions, rewards, dhats = [s], [], [], [torch.zeros(n, device=device)]
    term_sum = torch.zeros(len(REWARD_TERM_NAMES), device=device)
    fused = step_fn.policy is not None
    hist_t = torch.stack(hist, 0)
    for t in range(horizon):
        if fused:
            a, hist_t, s, r, term_vec, d = step_fn.policy_step(
                hist_t, goals_all[t], s, a_prev, start.xi, ref
            )
        else:
            a = policy(flatten_history_term_major(hist_t), goals_all[t])
            s, r, term_vec, logits, obs_clean = step_fn(s, a, a_prev, start.xi, ref)
            d = torch.sigmoid(logits)
            hist_t = torch.cat([hist_t[1:], add_noise(obs_clean)[None]], dim=0)
        term_sum += term_vec.detach().mean(-1)
        ref = ref_at(t + 1)
        a_prev = a
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
    dhat = torch.stack(dhats)
    states_t = torch.stack(states)
    e_root = wm.heads.root_error_features(
        states_t,
        ReferenceFrame(
            ref_all.body_pos_w,
            ref_all.body_quat_w,
            ref_all.body_lin_vel_w,
            ref_all.body_ang_vel_w,
            ref_all.root_pos_w.reshape(horizon + 1, n, 3),
            ref_all.root_quat_w.reshape(horizon + 1, n, 4),
        ),
    )
    return ImagRollout(
        states=states_t,
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=dhat,
        survival=torch.cumprod(1.0 - dhat, dim=0),  # c_0 = 1, c_t = c_{t-1} (1 - d_t)
        goals=goals_all,
        e_root=e_root,
        reward_terms={k: float(v / horizon) for k, v in zip(REWARD_TERM_NAMES, term_sum.tolist())},
    )


def policy_objective(
    ro: ImagRollout, terminal_values: torch.Tensor | None, gamma: float
) -> torch.Tensor:
    """J = mean_n [ sum_t gamma^t c_t r_{t+1} + gamma^H c_H V(s_H) ] (SHAC-style)."""
    H = ro.rewards.shape[0]
    disc = gamma ** torch.arange(H, device=ro.rewards.device, dtype=ro.rewards.dtype)
    J = (disc[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
    if terminal_values is not None:
        J = J + (gamma**H) * ro.survival[-1] * terminal_values
    return J.mean()


@torch.no_grad()
def gae_targets(
    rewards: torch.Tensor, dhat_next: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """R_t = r_{t+1} + gamma (1 - d_{t+1}) [ (1-lam) V_{t+1} + lam R_{t+1} ],  R_H = V_H."""
    H = rewards.shape[0]
    R = values[H].clone()
    out = torch.empty_like(rewards)
    for t in range(H - 1, -1, -1):
        R = rewards[t] + gamma * (1.0 - dhat_next[t]) * ((1.0 - lam) * values[t + 1] + lam * R)
        out[t] = R
    return out
