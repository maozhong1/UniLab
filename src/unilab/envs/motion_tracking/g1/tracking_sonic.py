"""G1 SONIC motion-tracking env (task ``G1SonicMotionTracking``) — task-3 obs layout.

Produces the sonic G1-path observation the ``SonicG1ActorModel`` consumes, as a
single 1570-dim actor stream ``obs`` = encoder-input(640) ++ proprio(930):

    encoder-input (640, sonic ``g1`` encoder):
        the command preserves SONIC's training-time reshape bug: concatenate flattened
        [all dof_pos frames, all dof_vel frames], reshape directly to (10, 58), then append
        anchor_ori6 per row. This is not semantic frame-major [q_t, dq_t], but ``last.pt``
        and the released encoder ONNX require the legacy layout.
    anchor 6D = first-2-columns of conj(robot_pelvis)*ref_root [m00,m01,m10,m11,m20,m21].

  proprio (930, sonic ``g1_dyn`` decoder tail, per-term history OLDEST-first, step1 ×10):
    his_base_angular_velocity (30 = PELVIS gyro 3 × 10)     [not torso — sonic base=root]
    his_body_joint_positions  (290 = joint_pos_rel 29 × 10)
    his_body_joint_velocities (290 = dof_vel 29 × 10)
    his_last_actions          (290 = last_actions 29 × 10)
    his_gravity_dir           (30 = projected gravity 3 × 10, pelvis frame)

The model splits 1570 → (640, 930) internally (``proprio_group=null``), so no
wrapper/obs_groups plumbing changes are needed. The critic stays the stock
286-dim BeyondMimic critic (value net must not be quantized).

Anchor body = ``pelvis`` (sonic convention) vs the stock ``torso_link``.

History buffer + reset semantics mirror the proven ``G1WBTObs`` SAC variant
(``tracking_obs.py``): on reset (``info["env_ids"]`` set, rows pre-subset by the
DR provider) all H slots are filled with the current value; on step the ring is
shifted (oldest out, current in). Future frames are clamped to the clip end so
lookahead never crosses a clip boundary.

Byte-compat status (verified against gear_sonic config/code + deploy ONNX registry):
MATCHED — proprio term order, oldest-first history, step5 command/anchor + step1
proprio, anchor 6D convention, gravity_dir pelvis-frame, joint_pos_rel. STILL
MUST-VERIFY NUMERICALLY — the full-29 joint order (sonic ONNX = isaaclab dof order;
UniLab motion NPZ = MuJoCo dof order; deploy has explicit mujoco→isaaclab remaps for
subsets). If those orders differ, a per-joint permutation is needed before warm-start
from last.pt. Until confirmed, prefer training from scratch. See byte-compat notes doc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.base import ControlConfig, Sensor
from unilab.utils.geometry import np_write_relative_anchor_transform_pos_rot6d
from unilab.utils.rotation import np_quat_apply_inverse

from ..common.config import MotionTrackingCfg
from ..common.rewards import RewardConfig
from ..common.tracking import MotionTrackingEnv

# sonic g1 encoder / decoder dims (see sonic_g1_core.py / sonic_onnx_interface_spec.md)
_CMD_PER_FRAME = 2  # joint_pos + joint_vel (× n_action)
_ANCHOR_ORI6 = 6
_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float32)

# --- MuJoCo → IsaacLab 29-dof joint permutation -----------------------------
# sonic's encoder/decoder ONNX consume joints in ISAACLAB order; UniLab motion
# NPZ + get_dof_pos are in MuJoCo (URDF kinematic-tree) order, and the two differ
# (not identity). Authoritative source: gear_sonic_deploy .../policy_parameters.hpp
# `mujoco_to_isaaclab` ("isaaclab order in mujoco index"): for isaaclab position i,
# the value is the mujoco index to read. So the isaaclab-order vector is built as
#   v_isaaclab = v_mujoco[_MUJOCO_TO_ISAACLAB].
# Deploy consumes isaaclab order for BOTH the encoder command (full-body motion
# joints, g1_deploy_onnx_ref.cpp:768 JointPositions() is stored isaaclab-order)
# AND the decoder proprio (jpos/jvel/last_actions, cpp:2827 body_q[i] = hw[m2i[i]];
# action out cpp:3120 uses isaaclab_to_mujoco). gyro / gravity_dir (base-frame 3-vec)
# and anchor_ori (6D) are NOT joint-indexed and must NOT be permuted.
_MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.intp,
)
# inverse (isaaclab → mujoco), kept for the action-output remap done by the policy
# integration layer (decoder emits isaaclab-order action; env applies mujoco order).
_ISAACLAB_TO_MUJOCO = np.empty(29, dtype=np.intp)
_ISAACLAB_TO_MUJOCO[_MUJOCO_TO_ISAACLAB] = np.arange(29, dtype=np.intp)

# SONIC trains normalized residual actions with scale 0.25 * effort_limit / Kp.
# Values are in MuJoCo actuator order because the action permutation is applied
# before the base environment converts actions to position targets.
_SONIC_ACTION_SCALE_MUJOCO = np.array(
    [
        0.25 * 139.0 / 99.098, 0.25 * 139.0 / 99.098, 0.25 * 88.0 / 40.179,
        0.25 * 139.0 / 99.098, 0.25 * 50.0 / 28.501, 0.25 * 50.0 / 28.501,
        0.25 * 139.0 / 99.098, 0.25 * 139.0 / 99.098, 0.25 * 88.0 / 40.179,
        0.25 * 139.0 / 99.098, 0.25 * 50.0 / 28.501, 0.25 * 50.0 / 28.501,
        0.25 * 88.0 / 40.179, 0.25 * 50.0 / 28.501, 0.25 * 50.0 / 28.501,
        0.25 * 25.0 / 14.251, 0.25 * 25.0 / 14.251, 0.25 * 25.0 / 14.251,
        0.25 * 25.0 / 14.251, 0.25 * 25.0 / 14.251, 0.25 * 5.0 / 16.778,
        0.25 * 5.0 / 16.778, 0.25 * 25.0 / 14.251, 0.25 * 25.0 / 14.251,
        0.25 * 25.0 / 14.251, 0.25 * 25.0 / 14.251, 0.25 * 25.0 / 14.251,
        0.25 * 5.0 / 16.778, 0.25 * 5.0 / 16.778,
    ],
    dtype=np.float32,
)


@dataclass
class SonicRewardConfig(RewardConfig):
    """SONIC-specific tracking terms absent from the generic task."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            **RewardConfig().scales,
            "motion_local_points": 2.0,
            "undesired_contacts": -0.1,
        }
    )
    std_local_points: float = 0.1


def _pack_sonic_encoder_command(joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    """Reproduce SONIC's legacy ``cat(...).reshape(F, 2*n)`` command layout."""
    rows, future_frames, num_joints = joint_pos.shape
    return np.concatenate(
        [joint_pos.reshape(rows, -1), joint_vel.reshape(rows, -1)], axis=1
    ).reshape(rows, future_frames, _CMD_PER_FRAME * num_joints)


@registry.envcfg("G1SonicMotionTracking")
@dataclass
class G1SonicMotionTrackingCfg(MotionTrackingCfg):
    """SONIC G1 tracking: pelvis anchor, multi-future command, proprio history.

    Byte-compat with sonic ``last.pt`` (verified vs gear_sonic + deploy ONNX registry):
    command_multi_future + anchor_ori_mf use ``step5`` (dt_future_ref_frames=0.1s @
    target_fps=50 → frame_skips=5); proprio history is step1 (per-control-step). See
    the byte-compat notes doc. base_ang_vel must be the PELVIS gyro (sonic uses
    root/base angular velocity), not the default torso_gyro.
    """

    anchor_body_name: str = "pelvis"
    num_future_frames: int = 10      # command / anchor-ori lookahead horizon
    future_stride: int = 5           # step5: 0.1s spacing @ 50fps (sonic frame_skips)
    proprio_history_len: int = 10    # per-term proprio history depth (step1)
    # Reorder every 29-dof vector the encoder/decoder consumes (command jpos/jvel,
    # proprio jpos_rel/dof_vel/last_actions) from UniLab MuJoCo order to sonic's
    # IsaacLab order. REQUIRED for last.pt warm-start / deploy-ONNX reuse; for
    # train-from-scratch it can be False (the net learns whatever self-consistent
    # order we feed, provided the action output is applied in that same order).
    mujoco_to_isaaclab_perm: bool = True
    # Companion of the above for the ACTION side: the sonic decoder emits actions in
    # IsaacLab joint order, but the env applies ctrl in MuJoCo order. When True, the
    # policy action is remapped IsaacLab→MuJoCo before apply_action so a warm-started
    # decoder drives the correct joints (and last_actions, stored MuJoCo-order then
    # re-permuted MuJoCo→IsaacLab in the obs, round-trips back to what the decoder
    # emitted). MUST be True whenever mujoco_to_isaaclab_perm is True AND weights are
    # warm-started from last.pt (or exported for the deploy ONNX). Default False so a
    # fresh-init run stays self-consistent without touching the action interface.
    action_output_isaaclab_to_mujoco: bool = False
    control_config: ControlConfig = field(
        default_factory=lambda: ControlConfig(action_scale=_SONIC_ACTION_SCALE_MUJOCO.copy())
    )
    reward_config: SonicRewardConfig = field(default_factory=SonicRewardConfig)
    # The base termination applies this permissive bound. The subclass below adds
    # SONIC's strict 0.15 m threshold except during low-reference motions.
    anchor_pos_z_threshold: float = 0.75
    ee_body_pos_z_threshold: float = 0.75
    strict_height_threshold: float = 0.15
    low_reference_height: float = 0.5
    low_reference_height_threshold: float = 0.75
    # Full pelvis orientation squared-angle threshold and ankle world-position limit.
    strict_anchor_ori_error_sq: float = 0.2
    strict_foot_pos_threshold: float = 0.2
    strict_local_point_body_names: tuple[str, ...] = (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )
    # base_ang_vel = pelvis gyro (sonic root ang-vel), not torso. pelvis_local_linvel
    # is already the pelvis sensor in the stock cfg; only gyro needs switching.
    sensor: Sensor = field(default_factory=lambda: Sensor(gyro="pelvis_gyro"))


@registry.env("G1SonicMotionTracking", sim_backend="mujoco")
@registry.env("G1SonicMotionTracking", sim_backend="motrix")
class G1SonicMotionTrackingEnv(MotionTrackingEnv):
    """G1 motion tracking with the sonic multi-future + history actor obs layout."""

    _cfg: G1SonicMotionTrackingCfg

    def __init__(self, cfg: G1SonicMotionTrackingCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._strict_local_point_indices = np.asarray(
            [cfg.body_names.index(name) for name in cfg.strict_local_point_body_names], dtype=np.intp
        )
        self._strict_foot_indices = np.asarray(
            [
                cfg.body_names.index("left_ankle_roll_link"),
                cfg.body_names.index("right_ankle_roll_link"),
            ],
            dtype=np.intp,
        )
        self._strict_point_error = np.empty(
            (num_envs, self._strict_local_point_indices.size, 3), dtype=get_global_dtype()
        )
        self._strict_point_reference = np.empty_like(self._strict_point_error)
        self._strict_point_rot6d = np.empty(
            (num_envs * self._strict_local_point_indices.size, 6), dtype=get_global_dtype()
        )
        self._strict_done = np.empty((num_envs,), dtype=bool)
        self._strict_ee_mask = np.empty((num_envs, self.ee_body_indices.size), dtype=bool)
        n = self._num_action
        self._F = int(cfg.num_future_frames)
        self._stride = int(cfg.future_stride)
        self._H = int(cfg.proprio_history_len)

        # MuJoCo→IsaacLab joint permutation (identity when disabled / n≠29)
        self._jperm = (
            _MUJOCO_TO_ISAACLAB
            if (cfg.mujoco_to_isaaclab_perm and n == 29)
            else np.arange(n, dtype=np.intp)
        )
        # Action-side IsaacLab→MuJoCo remap (None = pass actions through unchanged)
        self._action_out_perm = (
            _ISAACLAB_TO_MUJOCO
            if (cfg.action_output_isaaclab_to_mujoco and n == 29)
            else None
        )

        self._enc_dim = (_CMD_PER_FRAME * n + _ANCHOR_ORI6) * self._F      # 640 (n=29,F=10)
        self._proprio_frame_dim = 3 + n + n + n + 3                         # 93
        self._proprio_dim = self._proprio_frame_dim * self._H              # 930
        self._sonic_actor_dim = self._enc_dim + self._proprio_dim          # 1570

        dtype = get_global_dtype()
        H = self._H
        # per-term proprio history, oldest-first (mirrors G1WBTObs)
        self._hist: dict[str, np.ndarray] = {
            "gyro": np.zeros((num_envs, H, 3), dtype=dtype),
            "joint_pos_rel": np.zeros((num_envs, H, n), dtype=dtype),
            "dof_vel": np.zeros((num_envs, H, n), dtype=dtype),
            "last_actions": np.zeros((num_envs, H, n), dtype=dtype),
            "gravity_dir": np.zeros((num_envs, H, 3), dtype=dtype),
        }

    # obs_groups_spec reports the sonic actor width; the stock 160-dim actor built
    # by super()._compute_obs is discarded (we override "obs"). _actor_obs_dim is
    # left at the stock value so super's internal allocation stays correct.
    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._sonic_actor_dim, "critic": self._critic_obs_width}

    # -- future-frame gather ------------------------------------------------
    def _gather_future(self, env_ids: np.ndarray | None):
        """Return (joint_pos, joint_vel, body_pos_w, body_quat_w) each (R, F, ...)."""
        frames = self.motion_sampler.current_frames
        clip_end = self.motion_sampler.current_clip_end_frames
        if env_ids is not None:
            frames = frames[env_ids]
            clip_end = clip_end[env_ids]
        R, F = frames.shape[0], self._F
        offsets = np.arange(F, dtype=np.int32) * self._stride  # (F,)
        idx = np.minimum(frames[:, None] + offsets[None, :], clip_end[:, None])  # (R,F)
        md = self.motion_loader.get_motion_at_frame(idx.reshape(-1))
        jp = md.joint_pos.reshape(R, F, -1)
        jv = md.joint_vel.reshape(R, F, -1)
        bp = md.body_pos_w.reshape(R, F, -1, 3)
        bq = md.body_quat_w.reshape(R, F, -1, 4)
        # MuJoCo→IsaacLab: reorder the command joint channels the g1 encoder reads.
        # (body_pos/quat feed the anchor 6D, which is not joint-indexed — leave as-is.)
        if jp.shape[-1] == self._jperm.shape[0]:
            jp = jp[..., self._jperm]
            jv = jv[..., self._jperm]
        return jp, jv, bp, bq

    def _build_sonic_actor(
        self,
        info: dict,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        gyro: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> np.ndarray:
        dtype = get_global_dtype()
        n = self._num_action
        F = self._F
        env_ids = info.get("env_ids")
        is_reset = env_ids is not None
        R = dof_pos.shape[0]
        ai = self.anchor_body_idx

        # ---- encoder input: multi-future command + anchor-ori -------------
        jp, jv, fut_bp, fut_bq = self._gather_future(env_ids)  # (R,F,n) / (R,F,nb,3|4)

        robot_anchor_pos = robot_body_pos_w[:, ai]     # (R,3)
        robot_anchor_quat = robot_body_quat_w[:, ai]   # (R,4)
        src_pos = np.repeat(robot_anchor_pos, F, axis=0)   # (R*F,3) row-major matches reshape
        src_quat = np.repeat(robot_anchor_quat, F, axis=0)
        tgt_pos = fut_bp[:, :, ai].reshape(R * F, 3)
        tgt_quat = fut_bq[:, :, ai].reshape(R * F, 4)
        out_pos = np.empty((R * F, 3), dtype=dtype)
        out_rot6 = np.empty((R * F, 6), dtype=dtype)
        np_write_relative_anchor_transform_pos_rot6d(
            src_pos, src_quat, tgt_pos, tgt_quat, out_pos, out_rot6
        )
        anchor_mf = out_rot6.reshape(R, F, _ANCHOR_ORI6)  # (R,F,6)
        # Preserve SONIC's training-time temporal packing bug. Its command term first
        # concatenates flattened [all q frames, all dq frames], then reshapes that
        # 580-vector to (F, 58) before appending anchor orientation. Although this is
        # not frame-wise [q_t, dq_t], last.pt was trained against exactly this layout.
        command = _pack_sonic_encoder_command(jp, jv)
        per_frame = np.concatenate([command, anchor_mf], axis=2)  # (R,F,64)
        enc = per_frame.reshape(R, F * (_CMD_PER_FRAME * n + _ANCHOR_ORI6))  # 640

        # ---- proprio current-frame terms ----------------------------------
        bias = info.get("default_dof_pos_bias")
        effective_default = self.default_angles + bias if bias is not None else self.default_angles
        joint_pos_rel = np.asarray(dof_pos - effective_default, dtype=dtype)
        last_actions = info.get("current_actions")
        if not isinstance(last_actions, np.ndarray):
            last_actions = np.zeros((R, n), dtype=dtype)
        last_actions = np.asarray(last_actions, dtype=dtype)
        pelvis_quat = robot_body_quat_w[:, ai]  # anchor==pelvis for sonic
        gravity_dir = np_quat_apply_inverse(
            pelvis_quat, np.broadcast_to(_GRAVITY, (R, 3)).astype(pelvis_quat.dtype)
        ).astype(dtype)

        # MuJoCo→IsaacLab: reorder the three joint-indexed proprio terms the g1_dyn
        # decoder reads. gyro & gravity_dir are base-frame 3-vectors — never permuted.
        # (rel-subtraction commutes with the permutation since default_angles and
        # dof_pos are both MuJoCo-order here; we permute the resulting rel vector.)
        jp_perm = self._jperm
        if jp_perm.shape[0] == n:
            joint_pos_rel = joint_pos_rel[:, jp_perm]
            dof_vel = np.asarray(dof_vel, dtype=dtype)[:, jp_perm]
            last_actions = last_actions[:, jp_perm]
        else:
            dof_vel = np.asarray(dof_vel, dtype=dtype)

        components = {
            "gyro": np.asarray(gyro, dtype=dtype),
            "joint_pos_rel": joint_pos_rel,
            "dof_vel": dof_vel,
            "last_actions": last_actions,
            "gravity_dir": gravity_dir,
        }
        if is_reset:
            self._fill_history(env_ids, components)
        else:
            self._push_history(env_ids, components)

        # ---- proprio: per-term history flattened oldest-first --------------
        sel = slice(None) if env_ids is None else env_ids
        proprio_terms = [
            self._hist[k][sel].reshape(R, -1)
            for k in ("gyro", "joint_pos_rel", "dof_vel", "last_actions", "gravity_dir")
        ]
        proprio = np.concatenate(proprio_terms, axis=1)  # 930
        return np.concatenate([enc, proprio], axis=1, dtype=dtype)  # 1570

    def _push_history(self, env_ids: np.ndarray | None, components: dict[str, np.ndarray]) -> None:
        sel = slice(None) if env_ids is None else env_ids
        for key, val in components.items():
            buf = self._hist[key]
            buf[sel, :-1] = buf[sel, 1:]
            buf[sel, -1] = val

    def _fill_history(self, env_ids: np.ndarray | None, components: dict[str, np.ndarray]) -> None:
        sel = slice(None) if env_ids is None else env_ids
        for key, val in components.items():
            self._hist[key][sel, :] = val[:, None, :]

    def apply_action(self, actions: np.ndarray, state: Any) -> np.ndarray:
        """Remap the sonic decoder's IsaacLab-order action to MuJoCo order before the
        base PD conversion. With the remap on, ``current_actions`` (used for the
        last_actions obs term) is stored MuJoCo-order and re-permuted MuJoCo→IsaacLab
        in ``_build_sonic_actor`` — round-tripping back to the decoder's own order."""
        if self._action_out_perm is not None:
            actions = actions[:, self._action_out_perm]
        return super().apply_action(actions, state)

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns["motion_local_points"] = self._reward_motion_local_points

    def _reward_motion_local_points(self, ctx: Any) -> np.ndarray:
        """Track wrists and ankles relative to each motion's pelvis frame."""
        ref_pos = ctx.motion_data.body_pos_w[:, self._strict_local_point_indices]
        robot_pos = ctx.robot_body_pos_w[:, self._strict_local_point_indices]
        ref_anchor = ctx.motion_data.body_pos_w[:, self.anchor_body_idx]
        robot_anchor = ctx.robot_body_pos_w[:, self.anchor_body_idx]
        ref_quat = ctx.motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_quat = ctx.robot_body_quat_w[:, self.anchor_body_idx]

        point_count = self._strict_local_point_indices.size
        ref_anchor_tiled = np.broadcast_to(
            ref_anchor[:, None, :], (self._num_envs, point_count, 3)
        ).reshape(-1, 3)
        ref_quat_tiled = np.broadcast_to(
            ref_quat[:, None, :], (self._num_envs, point_count, 4)
        ).reshape(-1, 4)
        robot_anchor_tiled = np.broadcast_to(
            robot_anchor[:, None, :], (self._num_envs, point_count, 3)
        ).reshape(-1, 3)
        robot_quat_tiled = np.broadcast_to(
            robot_quat[:, None, :], (self._num_envs, point_count, 4)
        ).reshape(-1, 4)

        # Write both point clouds in their respective pelvis frames, then reuse
        # the robot buffer for the local-position error.
        np_write_relative_anchor_transform_pos_rot6d(
            ref_anchor_tiled,
            ref_quat_tiled,
            ref_pos.reshape(-1, 3),
            ref_quat_tiled,
            self._strict_point_reference.reshape(-1, 3),
            self._strict_point_rot6d,
        )
        np_write_relative_anchor_transform_pos_rot6d(
            robot_anchor_tiled,
            robot_quat_tiled,
            robot_pos.reshape(-1, 3),
            robot_quat_tiled,
            self._strict_point_error.reshape(-1, 3),
            self._strict_point_rot6d,
        )
        self._strict_point_error -= self._strict_point_reference
        np.square(self._strict_point_error, out=self._strict_point_error)
        np.sum(self._strict_point_error, axis=(1, 2), out=ctx.env_error)
        ctx.env_error /= self._strict_local_point_indices.size
        np.divide(ctx.env_error, -(self._cfg.reward_config.std_local_points**2), out=ctx.reward_term)
        np.exp(ctx.reward_term, out=ctx.reward_term)
        return ctx.reward_term

    def _compute_terminations(
        self,
        motion_data: Any,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> np.ndarray:
        terminated = super()._compute_terminations(motion_data, robot_body_pos_w, robot_body_quat_w)
        ref_anchor_pos = motion_data.body_pos_w[:, self.anchor_body_idx]
        robot_anchor_pos = robot_body_pos_w[:, self.anchor_body_idx]
        np.subtract(ref_anchor_pos[:, 2], robot_anchor_pos[:, 2], out=self._env_error)
        np.abs(self._env_error, out=self._env_error)
        np.greater(self._env_error, self._cfg.strict_height_threshold, out=self._env_bool)
        low_reference = ref_anchor_pos[:, 2] < self._cfg.low_reference_height
        np.greater(
            self._env_error,
            self._cfg.low_reference_height_threshold,
            out=self._strict_done,
        )
        self._env_bool[low_reference] = self._strict_done[low_reference]
        terminated |= self._env_bool

        if self._has_ee_body_indices:
            np.subtract(
                self.body_pos_relative_w[:, self.ee_body_indices, 2],
                robot_body_pos_w[:, self.ee_body_indices, 2],
                out=self._ee_pos_error_z,
            )
            np.abs(self._ee_pos_error_z, out=self._ee_pos_error_z)
            np.greater(
                self._ee_pos_error_z,
                self._cfg.strict_height_threshold,
                out=self._ee_terminated,
            )
            np.greater(
                self._ee_pos_error_z,
                self._cfg.low_reference_height_threshold,
                out=self._strict_ee_mask,
            )
            self._ee_terminated[low_reference] = self._strict_ee_mask[low_reference]
            np.logical_or.reduce(self._ee_terminated, axis=1, out=self._env_bool)
            terminated |= self._env_bool

        ref_quat = motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_quat = robot_body_quat_w[:, self.anchor_body_idx]
        np.sum(ref_quat * robot_quat, axis=1, out=self._env_error)
        np.abs(self._env_error, out=self._env_error)
        np.clip(self._env_error, 0.0, 1.0, out=self._env_error)
        np.arccos(self._env_error, out=self._env_error)
        self._env_error *= 2.0
        np.square(self._env_error, out=self._env_error)
        np.greater(self._env_error, self._cfg.strict_anchor_ori_error_sq, out=self._env_bool)
        terminated |= self._env_bool

        np.subtract(
            self.body_pos_relative_w[:, self._strict_foot_indices],
            robot_body_pos_w[:, self._strict_foot_indices],
            out=self._strict_point_error[:, : self._strict_foot_indices.size],
        )
        np.square(self._strict_point_error[:, : self._strict_foot_indices.size], out=self._strict_point_error[:, : self._strict_foot_indices.size])
        np.sum(self._strict_point_error[:, : self._strict_foot_indices.size], axis=2, out=self._ee_pos_error_z[:, : self._strict_foot_indices.size])
        np.sqrt(self._ee_pos_error_z[:, : self._strict_foot_indices.size], out=self._ee_pos_error_z[:, : self._strict_foot_indices.size])
        np.greater(self._ee_pos_error_z[:, : self._strict_foot_indices.size], self._cfg.strict_foot_pos_threshold, out=self._ee_terminated[:, : self._strict_foot_indices.size])
        np.logical_or.reduce(self._ee_terminated[:, : self._strict_foot_indices.size], axis=1, out=self._env_bool)
        terminated |= self._env_bool
        return terminated

    def _compute_obs(
        self,
        info: dict,
        motion_data: Any,
        linvel: np.ndarray,
        gyro: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        robot_body_pos_w: np.ndarray,
        robot_body_quat_w: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # Reuse the stock path for the (unchanged) 286-dim critic; discard its
        # 160-dim actor and replace with the sonic 1570-dim actor stream.
        base = super()._compute_obs(
            info, motion_data, linvel, gyro, dof_pos, dof_vel, robot_body_pos_w, robot_body_quat_w
        )
        actor = self._build_sonic_actor(
            info, dof_pos, dof_vel, gyro, robot_body_pos_w, robot_body_quat_w
        )
        return {"obs": actor, "critic": base["critic"]}
