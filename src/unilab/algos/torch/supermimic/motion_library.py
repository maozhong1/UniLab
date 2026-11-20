"""Compact reference library on the learner device (imagination side).

Per frame only the (13+2N)-d reference state
``[root_pos, root_quat, q, root_lin_vel_w, root_ang_vel_w, qd]`` is stored; tracked-body states
and goal features are computed on the fly with the differentiable FK. Frames are indexed with
the same global frame axis as the env's ``SuperMimicMotionLibrary`` (same clip order), so the
target frames logged in the replay buffer address this library directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .heads import ReferenceFrame
from .kinematics import HumanoidKinematics
from .rotations import projected_gravity, quat_apply_inverse, yaw_quat
from .state_spec import split_state


@dataclass
class RefState:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    joint_pos: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor  # world frame
    joint_vel: torch.Tensor


def goal_features(ref: RefState, default_joint_pos: torch.Tensor) -> torch.Tensor:
    """(..., 2N+10) goal features (same layout as the env's ``goal`` group)."""
    q = ref.root_quat_w
    return torch.cat(
        [
            ref.joint_pos - default_joint_pos,
            ref.joint_vel,
            projected_gravity(q),
            quat_apply_inverse(q, ref.root_ang_vel_w),
            quat_apply_inverse(yaw_quat(q), ref.root_lin_vel_w),
            ref.root_pos_w[..., 2:3],
        ],
        dim=-1,
    )


class MotionLibrary:
    def __init__(
        self,
        ref_states: np.ndarray,
        clip_start: np.ndarray,
        clip_len: np.ndarray,
        default_joint_pos: np.ndarray,
        fk: HumanoidKinematics,
        tracked_bodies: tuple[str, ...],
        device: str | torch.device,
        store_dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = torch.float32
        self._ref = torch.as_tensor(ref_states, device=self.device).to(store_dtype)
        self.num_frames = int(self._ref.shape[0])
        self.default_joint_pos = torch.as_tensor(
            np.asarray(default_joint_pos), dtype=self.dtype, device=self.device
        )
        self.goal_feat_dim = 2 * self.default_joint_pos.numel() + 10
        self.clip_start = torch.as_tensor(clip_start, dtype=torch.long, device=self.device)
        self.clip_len = torch.as_tensor(clip_len, dtype=torch.long, device=self.device)
        self.clip_last = self.clip_start + self.clip_len - 1
        self.frame_clip = torch.repeat_interleave(
            torch.arange(len(clip_start), device=self.device), self.clip_len
        )
        self._fk = fk
        self._tracked_idx = fk.body_indices(tracked_bodies)

    @classmethod
    def from_env_library(
        cls, lib, fk: HumanoidKinematics, tracked_bodies, device, store_dtype=torch.float32
    ) -> MotionLibrary:
        return cls(
            lib.ref_states(),
            lib.clip_start,
            lib.clip_len,
            lib.default_joint_pos,
            fk,
            tracked_bodies,
            device,
            store_dtype,
        )

    def ref(self, frame: torch.Tensor) -> RefState:
        return RefState(*split_state(self._ref[frame].to(self.dtype)))

    def last_frame_of(self, frame: torch.Tensor) -> torch.Tensor:
        return self.clip_last[self.frame_clip[frame]]

    def ref_qpos_qvel(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.ref(frame)
        qpos = torch.cat([r.root_pos_w, r.root_quat_w, r.joint_pos], dim=-1)
        qvel = torch.cat(
            [r.root_lin_vel_w, quat_apply_inverse(r.root_quat_w, r.root_ang_vel_w), r.joint_vel],
            dim=-1,
        )
        return qpos, qvel

    @torch.no_grad()
    def reference_at(self, frame: torch.Tensor) -> ReferenceFrame:
        """Tracked-body world states (via FK) and root pose of the given frames."""
        r = self.ref(frame)
        qpos, qvel = self.ref_qpos_qvel(frame)
        out = self._fk(qpos, qvel)
        i = self._tracked_idx
        return ReferenceFrame(
            out["pos"][..., i, :],
            out["quat"][..., i, :],
            out["lin_vel"][..., i, :],
            out["ang_vel"][..., i, :],
            r.root_pos_w,
            r.root_quat_w,
        )

    def goal_window(self, frame: torch.Tensor, k: int) -> torch.Tensor:
        """(n, k*(2N+10)) goal features of frames frame..frame+k-1, clamped at the clip end."""
        idx = torch.minimum(
            frame[:, None] + torch.arange(k, device=self.device)[None, :],
            self.last_frame_of(frame)[:, None],
        )
        feat = goal_features(self.ref(idx.reshape(-1)), self.default_joint_pos)
        return feat.reshape(frame.shape[0], k * self.goal_feat_dim)
