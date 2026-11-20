"""Analytic heads of the world model: observation, tracking reward and termination features.

Everything is a pure differentiable function of the simulator state
``s = [qpos (7+N) | qvel (6+N)]``, the previous clean action, the sim context ``xi`` and the
reference frame, so imagined rollouts score predicted states exactly like the simulator does
(torch twin of ``unilab.envs.motion_tracking.h2.supermimic``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kinematics import HumanoidKinematics
from .rotations import (
    heading_yaw,
    projected_gravity,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    rotvec_from_quat,
    wrap_to_pi,
    yaw_quat,
)
from .state_spec import split_state

REWARD_TERM_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
)


@dataclass
class TrackingRewardCfg:
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    dt: float = 0.02

    @property
    def max_step_reward(self) -> float:
        return self.dt * (
            self.w_body_pos
            + self.w_body_ori
            + self.w_body_lin_vel
            + self.w_body_ang_vel
            + self.w_root_pos
            + self.w_root_ori
        )


def obs_term_dims(nj: int) -> tuple[int, ...]:
    """Widths of the observation terms (joint_pos, joint_vel, base_ang_vel, projected_gravity, actions)."""
    return (nj, nj, 3, 3, nj)


def analytic_observation(
    s: torch.Tensor, prev_action: torch.Tensor, xi: torch.Tensor, default_joint_pos: torch.Tensor
) -> torch.Tensor:
    """Noise-free (3N+6)-d proprioception; ``xi[..., 1:1+N]`` is the encoder bias."""
    _, root_quat, q, _, ang_b, qd = split_state(s)
    bias = xi[..., 1 : 1 + q.shape[-1]]
    return torch.cat(
        [q + bias - default_joint_pos, qd, ang_b, projected_gravity(root_quat), prev_action], dim=-1
    )


def flatten_history_term_major(hist: torch.Tensor) -> torch.Tensor:
    """(H, n, 3N+6) oldest -> newest  ->  (n, H*(3N+6)) term-major ``[A_t0..A_tH-1, B_t0..]``."""
    H, n, D = hist.shape
    chunks, start = [], 0
    for d in obs_term_dims((D - 6) // 3):
        chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(n, H * d))
        start += d
    return torch.cat(chunks, dim=-1)


@dataclass
class ReferenceFrame:
    """Reference quantities of one target frame for a batch."""

    body_pos_w: torch.Tensor | None  # (n, B, 3)
    body_quat_w: torch.Tensor | None  # (n, B, 4)
    body_lin_vel_w: torch.Tensor | None  # (n, B, 3)
    body_ang_vel_w: torch.Tensor | None  # (n, B, 3)
    root_pos_w: torch.Tensor  # (n, 3)
    root_quat_w: torch.Tensor  # (n, 4)


def reanchor_reference(
    ref_body_pos_w,
    ref_body_quat_w,
    ref_anchor_pos_w,
    ref_anchor_quat_w,
    robot_anchor_pos_w,
    robot_anchor_quat_w,
):
    """Reference bodies in a frame sharing the robot anchor's xy position and yaw (BeyondMimic)."""
    nb = ref_body_pos_w.shape[-2]
    delta_pos = torch.cat([robot_anchor_pos_w[..., :2], ref_anchor_pos_w[..., 2:3]], dim=-1)
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))
    delta_ori_b = delta_ori[..., None, :].expand(*delta_ori.shape[:-1], nb, 4)
    rel_quat = quat_mul(delta_ori_b, ref_body_quat_w)
    rel_pos = delta_pos[..., None, :] + quat_apply(
        delta_ori_b, ref_body_pos_w - ref_anchor_pos_w[..., None, :]
    )
    return rel_pos, rel_quat


def _sq_quat_err(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle, smooth at zero error."""
    v = rotvec_from_quat(quat_mul(q1, quat_inv(q2)))
    return (v * v).sum(-1)


def tracking_reward_terms(
    *,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    anchor_index: int,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,
    cfg: TrackingRewardCfg,
) -> dict[str, torch.Tensor]:
    """Per-term weighted, dt-scaled rewards (n,) plus ``total``."""
    ra_pos, ra_quat = (
        robot_body_pos_w[..., anchor_index, :],
        robot_body_quat_w[..., anchor_index, :],
    )
    ma_pos, ma_quat = ref_body_pos_w[..., anchor_index, :], ref_body_quat_w[..., anchor_index, :]
    rel_pos, rel_quat = reanchor_reference(
        ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat
    )

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)
    e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)
    e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w).mean(-1)
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)
    lo, hi = soft_limits[..., 0], soft_limits[..., 1]
    joint_limit = ((lo - joint_pos).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)).sum(-1)
    action_rate = ((action - prev_action) ** 2).sum(-1)

    dt = cfg.dt
    terms = {
        "motion_global_root_pos": dt
        * cfg.w_root_pos
        * torch.exp(-e_root_pos / cfg.std_root_pos**2),
        "motion_global_root_ori": dt
        * cfg.w_root_ori
        * torch.exp(-e_root_ori / cfg.std_root_ori**2),
        "motion_body_pos": dt * cfg.w_body_pos * torch.exp(-e_body_pos / cfg.std_body_pos**2),
        "motion_body_ori": dt * cfg.w_body_ori * torch.exp(-e_body_ori / cfg.std_body_ori**2),
        "motion_body_lin_vel": dt
        * cfg.w_body_lin_vel
        * torch.exp(-e_body_lin / cfg.std_body_lin_vel**2),
        "motion_body_ang_vel": dt
        * cfg.w_body_ang_vel
        * torch.exp(-e_body_ang / cfg.std_body_ang_vel**2),
        "action_rate_l2": dt * cfg.w_action_rate * action_rate,
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,
    }
    total = terms["motion_global_root_pos"]
    for k in REWARD_TERM_NAMES[1:]:
        total = total + terms[k]
    terms["total"] = total
    return terms


class AnalyticHeads:
    """Reward / termination features computed from a (predicted) state through differentiable FK."""

    def __init__(
        self,
        fk: HumanoidKinematics,
        tracked_body_names: tuple[str, ...],
        anchor_body_name: str,
        root_body_name: str,
        soft_limits: torch.Tensor,
        default_joint_pos: torch.Tensor,
        reward_cfg: TrackingRewardCfg | None = None,
    ):
        self.fk = fk
        self.tracked_idx = fk.body_indices(tracked_body_names)
        self.anchor_index = tracked_body_names.index(anchor_body_name)
        self.root_fk_index = fk.body_index(root_body_name)
        self.soft_limits = soft_limits
        self.default_joint_pos = default_joint_pos
        self.reward_cfg = reward_cfg or TrackingRewardCfg()

    def robot_bodies(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.fk(s[..., : self.fk.nq], s[..., self.fk.nq :])
        idx = self.tracked_idx
        return {
            "pos": out["pos"][..., idx, :],
            "quat": out["quat"][..., idx, :],
            "lin_vel": out["lin_vel"][..., idx, :],
            "ang_vel": out["ang_vel"][..., idx, :],
        }

    def reward(
        self,
        s_next: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        ref: ReferenceFrame,
        bodies=None,
    ) -> dict[str, torch.Tensor]:
        """Reward r_{t+1} of reaching ``s_next`` under the target frame of step t."""
        b = bodies if bodies is not None else self.robot_bodies(s_next)
        return tracking_reward_terms(
            robot_body_pos_w=b["pos"],
            robot_body_quat_w=b["quat"],
            robot_body_lin_vel_w=b["lin_vel"],
            robot_body_ang_vel_w=b["ang_vel"],
            ref_body_pos_w=ref.body_pos_w,
            ref_body_quat_w=ref.body_quat_w,
            ref_body_lin_vel_w=ref.body_lin_vel_w,
            ref_body_ang_vel_w=ref.body_ang_vel_w,
            anchor_index=self.anchor_index,
            action=action,
            prev_action=prev_action,
            joint_pos=s_next[..., 7 : self.fk.nq],
            soft_limits=self.soft_limits,
            cfg=self.reward_cfg,
        )

    @staticmethod
    def termination_features(
        root_pos: torch.Tensor, root_quat: torch.Tensor, ref: ReferenceFrame
    ) -> torch.Tensor:
        """[height error, tilt error] (n, 2): the quantities the env thresholds (0.25 m, 0.8)."""
        g = torch.tensor([0.0, 0.0, -1.0], dtype=root_quat.dtype, device=root_quat.device).expand(
            root_quat.shape[:-1] + (3,)
        )
        tilt = (
            quat_apply_inverse(ref.root_quat_w, g)[..., 2]
            - quat_apply_inverse(root_quat, g)[..., 2]
        )
        height = ref.root_pos_w[..., 2] - root_pos[..., 2]
        return torch.stack([height, tilt], dim=-1)

    @staticmethod
    def root_error_features(s: torch.Tensor, ref: ReferenceFrame) -> torch.Tensor:
        """Critic-only privileged drift features: root position error in the reference heading frame (3) and yaw error (1)."""
        dp = quat_apply_inverse(yaw_quat(ref.root_quat_w), s[..., 0:3] - ref.root_pos_w)
        dyaw = wrap_to_pi(heading_yaw(s[..., 3:7]) - heading_yaw(ref.root_quat_w))
        return torch.cat([dp, dyaw[..., None]], dim=-1)
