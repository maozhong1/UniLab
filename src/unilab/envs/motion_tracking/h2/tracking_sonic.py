"""H2 SONIC motion-tracking env (task ``H2SonicMotionTracking``) — full-train layout.

A Unitree H2 (31-DOF) clone of the G1 sonic tracking env (``g1/tracking_sonic.py``).
H2 = G1's 29 joints + ``head_pitch``/``head_yaw``; all other joint and body names are
identical, so anchor (``pelvis``), the 14 tracked ``body_names``, and the sonic obs
layout carry over unchanged — everything is derived from ``n = self._num_action`` (31):

    encoder-input (680 = (2*31 + 6) * 10):  multi-future command (jpos+jvel, sonic's
        legacy [all q, all dq]->reshape(F,2n) packing) ++ anchor 6D, F=10.
    proprio       (990 = (3 + 31 + 31 + 31 + 3) * 10):  pelvis gyro ++ joint_pos_rel ++
        dof_vel ++ last_actions ++ gravity_dir, per-term history oldest-first, H=10.
    actor total   (1670 = 680 + 990).  ``SonicH2ActorModel`` splits it internally.

From-scratch training only (no G1 ``last.pt`` in H2 joint layout): NO MuJoCo<->IsaacLab
permutation (both perms stay False; everything is one self-consistent MuJoCo joint
order). Critic uses the privileged_mf_hist path (1745 for n=31, nb=14), trained fresh.

Action scale = 0.25 * effort / Kp in MJCF actuator order, from gear_sonic h2.py
(reuses G1 PD/armature constants; effort ≈ 3x G1, plus a head group).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.base import ControlConfig, Sensor
from unilab.utils.geometry import np_write_relative_anchor_transform_pos_rot6d
from unilab.utils.rotation import np_quat_apply_inverse

from ..common.config import MotionTrackingCfg
from ..common.rewards import RewardConfig
from ..common.tracking import MotionTrackingEnv

_CMD_PER_FRAME = 2  # joint_pos + joint_vel (× n_action)
_ANCHOR_ORI6 = 6
_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float32)

# H2 SONIC residual-action scale = 0.25 * effort_limit / Kp, in H2 MJCF actuator order
# (L-leg, R-leg, waist, head, L-arm, R-arm). Kp/effort from gear_sonic h2.py; must match
# the <position kp forcerange> actuators authored in robots/h2/h2.xml.
_SONIC_ACTION_SCALE_MUJOCO = np.array(
    [
        # left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_roll, ankle_pitch
        0.25 * 417.0 / 99.0984, 0.25 * 417.0 / 99.0984, 0.25 * 264.0 / 40.1792,
        0.25 * 417.0 / 99.0984, 0.25 * 150.0 / 28.5012, 0.25 * 150.0 / 28.5012,
        # right leg
        0.25 * 417.0 / 99.0984, 0.25 * 417.0 / 99.0984, 0.25 * 264.0 / 40.1792,
        0.25 * 417.0 / 99.0984, 0.25 * 150.0 / 28.5012, 0.25 * 150.0 / 28.5012,
        # waist: yaw, roll, pitch
        0.25 * 264.0 / 40.1792, 0.25 * 150.0 / 28.5012, 0.25 * 150.0 / 28.5012,
        # head: pitch, yaw
        0.25 * 150.0 / 28.5012, 0.25 * 150.0 / 28.5012,
        # left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        0.25 * 75.0 / 14.2506, 0.25 * 75.0 / 14.2506, 0.25 * 75.0 / 14.2506,
        0.25 * 75.0 / 14.2506, 0.25 * 75.0 / 14.2506, 0.25 * 15.0 / 16.7783, 0.25 * 15.0 / 16.7783,
        # right arm
        0.25 * 75.0 / 14.2506, 0.25 * 75.0 / 14.2506, 0.25 * 75.0 / 14.2506,
        0.25 * 75.0 / 14.2506, 0.25 * 75.0 / 14.2506, 0.25 * 15.0 / 16.7783, 0.25 * 15.0 / 16.7783,
    ],
    dtype=np.float32,
)

# Official privileged_mf_hist critic tracks these 14 bodies (identical to G1 — all names
# exist in H2). Order fixed by gear_sonic commands/terms/motion.yaml body_names.
_CRITIC_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


def _pack_sonic_encoder_command(joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
    """Reproduce SONIC's legacy ``cat(...).reshape(F, 2*n)`` command layout."""
    rows, future_frames, num_joints = joint_pos.shape
    return np.concatenate(
        [joint_pos.reshape(rows, -1), joint_vel.reshape(rows, -1)], axis=1
    ).reshape(rows, future_frames, _CMD_PER_FRAME * num_joints)


@dataclass
class SonicRewardConfig(RewardConfig):
    """SONIC-specific tracking terms absent from the generic task."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            **RewardConfig().scales,
            "motion_local_points": 2.0,
            "undesired_contacts": -0.1,
            # Foot-slip / foot-yaw penalties. H2's single foot-mesh contact has no
            # torsional friction, so a support foot can free-spin on turns and wreck
            # motion_body_ang_vel tracking. These shape the *policy* (not the contact
            # model), so train and deploy stay on the identical h2.xml. Default OFF;
            # enable per-run via reward.scales.feet_slide / feet_yaw (negative=penalty).
            "feet_slide": 0.0,
            "feet_yaw": 0.0,
            # ── P0/P1 smoothness terms (see the matching _reward_* methods) ──
            # Ported from the official SONIC reward + the G1 sim2real variant and
            # retuned for H2 (taller/heavier, effort ≈ 3× G1). All in per-step
            # (pre-ctrl_dt) units; watch reward/<name> in the iteration log and
            # scale the weight so each penalty sits well below the tracking terms.
            #   joint_acc_l2      : official "feet_acc" — L2 of ankle joint accel.
            #                       Joint-space (rad/s²) so ~robot-size invariant;
            #                       keep the official -2.5e-7.
            #   joint_torque_l2   : L2 of PD torque. H2 effort ≈ 3× G1 → torque² ≈ 9×,
            #                       so start ~9× LOWER than a G1 weight. Tune via log
            #                       (target roughly -0.05 … -0.15).
            #   anti_shake_ang_vel: deadzone jitter penalty, wrists (body ang-vel) +
            #                       head (joint-vel). Official weight; H2 adds an
            #                       actuated head that otherwise has zero regulariser.
            "joint_acc_l2": -2.5e-7,
            "joint_torque_l2": -1.0e-6,
            "anti_shake_ang_vel": -5.0e-3,
        }
    )
    std_local_points: float = 0.1
    # Robot foot world-z below this (flat ground) counts as "in contact" for the
    # feet_slide / feet_yaw gates. Calibrate to H2 ankle_roll_link stance height.
    feet_contact_height: float = 0.15
    # Angular-speed / joint-speed deadzone (rad/s) for anti_shake_ang_vel. No penalty
    # below this, so intentional wrist/head motion is free; only jitter is punished.
    anti_shake_threshold: float = 1.5


@registry.envcfg("H2SonicMotionTracking")
@dataclass
class H2SonicMotionTrackingCfg(MotionTrackingCfg):
    """SONIC H2 tracking: pelvis anchor, multi-future command, proprio history.

    From-scratch full training: encoder + decoder both trainable, no warm-start, no
    joint permutation (single MuJoCo joint order). Critic = privileged_mf_hist (1745).
    """

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "h2" / "scene_flat.xml")
        )
    )
    motion_file: str | list[str] = str(ASSETS_ROOT_PATH / "motions" / "h2" / "stand.npz")

    anchor_body_name: str = "pelvis"
    num_future_frames: int = 10
    future_stride: int = 5
    proprio_history_len: int = 10
    # From-scratch: keep both perms OFF (single MuJoCo joint order).
    mujoco_to_isaaclab_perm: bool = False
    action_output_isaaclab_to_mujoco: bool = False
    control_config: ControlConfig = field(
        default_factory=lambda: ControlConfig(action_scale=_SONIC_ACTION_SCALE_MUJOCO.copy())
    )
    reward_config: SonicRewardConfig = field(default_factory=SonicRewardConfig)

    anchor_pos_z_threshold: float = 0.75
    ee_body_pos_z_threshold: float = 0.75
    strict_height_threshold: float = 0.30
    low_reference_height: float = 0.5
    low_reference_height_threshold: float = 0.75
    strict_anchor_ori_error_sq: float = 0.2
    strict_foot_pos_threshold: float = 0.2
    strict_local_point_body_names: tuple[str, ...] = (
        "left_wrist_yaw_link", "right_wrist_yaw_link",
        "left_ankle_roll_link", "right_ankle_roll_link",
    )
    sensor: Sensor = field(default_factory=lambda: Sensor(gyro="pelvis_gyro"))
    critic_include_future: bool = True
    # Enabled via +env.critic_privileged_mf_hist=true in full_train_h2.sh (1745-d critic).
    critic_privileged_mf_hist: bool = False


@registry.env("H2SonicMotionTracking", sim_backend="mujoco")
@registry.env("H2SonicMotionTracking", sim_backend="motrix")
class H2SonicMotionTrackingEnv(MotionTrackingEnv):
    """H2 motion tracking with the sonic multi-future + history actor obs layout."""

    _cfg: H2SonicMotionTrackingCfg

    def __init__(self, cfg: H2SonicMotionTrackingCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._strict_local_point_indices = np.asarray(
            [cfg.body_names.index(name) for name in cfg.strict_local_point_body_names], dtype=np.intp
        )
        self._strict_foot_indices = np.asarray(
            [cfg.body_names.index("left_ankle_roll_link"),
             cfg.body_names.index("right_ankle_roll_link")], dtype=np.intp,
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

        # feet_slide / feet_yaw penalty scratch (zero-alloc hot path). Contact is
        # proxied by robot foot world-z < feet_contact_height (see SonicRewardConfig).
        _fdt = get_global_dtype()
        _nf = self._strict_foot_indices.size
        self._foot_xy_sq = np.empty((num_envs, _nf, 2), dtype=_fdt)
        self._foot_perfoot = np.empty((num_envs, _nf), dtype=_fdt)
        self._foot_gate = np.empty((num_envs, _nf), dtype=bool)
        self._foot_penalty_slide = np.empty((num_envs,), dtype=_fdt)
        self._foot_penalty_yaw = np.empty((num_envs,), dtype=_fdt)

        n = self._num_action
        self._F = int(cfg.num_future_frames)
        self._stride = int(cfg.future_stride)
        self._H = int(cfg.proprio_history_len)

        # No permutation for H2 (from scratch, single MuJoCo joint order).
        self._jperm = np.arange(n, dtype=np.intp)
        self._action_out_perm = None

        self._enc_dim = (_CMD_PER_FRAME * n + _ANCHOR_ORI6) * self._F   # 680 (n=31,F=10)
        self._proprio_frame_dim = 3 + n + n + n + 3                     # 99
        self._proprio_dim = self._proprio_frame_dim * self._H          # 990
        self._sonic_actor_dim = self._enc_dim + self._proprio_dim      # 1670

        dtype = get_global_dtype()
        H = self._H
        self._hist: dict[str, np.ndarray] = {
            "gyro": np.zeros((num_envs, H, 3), dtype=dtype),
            "joint_pos_rel": np.zeros((num_envs, H, n), dtype=dtype),
            "dof_vel": np.zeros((num_envs, H, n), dtype=dtype),
            "last_actions": np.zeros((num_envs, H, n), dtype=dtype),
            "gravity_dir": np.zeros((num_envs, H, 3), dtype=dtype),
        }

        self._critic_body_indices = np.asarray(
            [cfg.body_names.index(name) for name in _CRITIC_BODY_NAMES], dtype=np.intp
        )
        nb = self._critic_body_indices.size
        self._critic_mf_hist_dim = (
            _CMD_PER_FRAME * n * self._F + 3 + _ANCHOR_ORI6 + 9 * nb + (6 + 3 * n) * self._H
        )
        self._chist: dict[str, np.ndarray] = {}
        if cfg.critic_privileged_mf_hist:
            self._chist = {
                "base_lin_vel": np.zeros((num_envs, H, 3), dtype=dtype),
                "base_ang_vel": np.zeros((num_envs, H, 3), dtype=dtype),
                "joint_pos_rel": np.zeros((num_envs, H, n), dtype=dtype),
                "joint_vel": np.zeros((num_envs, H, n), dtype=dtype),
                "last_actions": np.zeros((num_envs, H, n), dtype=dtype),
            }

        # ── P0/P1 smoothness terms: joint_acc_l2 / joint_torque_l2 / anti_shake ──
        # Joint order is the MJCF actuator order (no permutation for H2), so actuator
        # names, PD gains (kp/kd), dof_pos and dof_vel are all mutually aligned.
        act_names = self._backend.get_actuator_names()
        self._ankle_dof_indices = np.asarray(
            [i for i, nm in enumerate(act_names) if "ankle" in nm], dtype=np.intp
        )
        self._head_dof_indices = np.asarray(
            [i for i, nm in enumerate(act_names) if "head" in nm], dtype=np.intp
        )
        try:
            kp, kd = self._backend.get_actuator_gains()
            self._torque_kp = np.asarray(kp, dtype=dtype)
            self._torque_kd = np.asarray(kd, dtype=dtype)
        except Exception:  # backend without PD gains → joint_torque_l2 returns 0
            self._torque_kp = None
            self._torque_kd = None
        # Wrists ARE tracked bodies (14-body set); head is not, so anti_shake handles
        # the head in joint-velocity space instead (see _reward_anti_shake_ang_vel).
        self._wrist_body_indices = np.asarray(
            [cfg.body_names.index("left_wrist_yaw_link"),
             cfg.body_names.index("right_wrist_yaw_link")], dtype=np.intp,
        )
        # Previous-step joint velocity for the finite-difference joint acceleration.
        # Updated at the end of every _compute_obs (including reset rows) so the first
        # post-reset sample sees Δv from the new reference state, not a spurious jump.
        self._prev_dof_vel = np.zeros((num_envs, n), dtype=dtype)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        if self._cfg.critic_privileged_mf_hist:
            return {"obs": self._sonic_actor_dim, "critic": self._critic_mf_hist_dim}
        critic_width = self._critic_obs_width
        if self._cfg.critic_include_future:
            critic_width += self._enc_dim
        return {"obs": self._sonic_actor_dim, "critic": critic_width}

    def _gather_future(self, env_ids: np.ndarray | None):
        frames = self.motion_sampler.current_frames
        clip_end = self.motion_sampler.current_clip_end_frames
        if env_ids is not None:
            frames = frames[env_ids]
            clip_end = clip_end[env_ids]
        R, F = frames.shape[0], self._F
        offsets = np.arange(F, dtype=np.int32) * self._stride
        idx = np.minimum(frames[:, None] + offsets[None, :], clip_end[:, None])
        md = self.motion_loader.get_motion_at_frame(idx.reshape(-1))
        jp = md.joint_pos.reshape(R, F, -1)
        jv = md.joint_vel.reshape(R, F, -1)
        bp = md.body_pos_w.reshape(R, F, -1, 3)
        bq = md.body_quat_w.reshape(R, F, -1, 4)
        return jp, jv, bp, bq

    def _build_sonic_actor(
        self, info: dict, dof_pos: np.ndarray, dof_vel: np.ndarray, gyro: np.ndarray,
        robot_body_pos_w: np.ndarray, robot_body_quat_w: np.ndarray,
    ) -> np.ndarray:
        dtype = get_global_dtype()
        n = self._num_action
        F = self._F
        env_ids = info.get("env_ids")
        is_reset = env_ids is not None
        R = dof_pos.shape[0]
        ai = self.anchor_body_idx

        jp, jv, fut_bp, fut_bq = self._gather_future(env_ids)
        robot_anchor_pos = robot_body_pos_w[:, ai]
        robot_anchor_quat = robot_body_quat_w[:, ai]
        src_pos = np.repeat(robot_anchor_pos, F, axis=0)
        src_quat = np.repeat(robot_anchor_quat, F, axis=0)
        tgt_pos = fut_bp[:, :, ai].reshape(R * F, 3)
        tgt_quat = fut_bq[:, :, ai].reshape(R * F, 4)
        out_pos = np.empty((R * F, 3), dtype=dtype)
        out_rot6 = np.empty((R * F, 6), dtype=dtype)
        np_write_relative_anchor_transform_pos_rot6d(
            src_pos, src_quat, tgt_pos, tgt_quat, out_pos, out_rot6
        )
        anchor_mf = out_rot6.reshape(R, F, _ANCHOR_ORI6)
        command = _pack_sonic_encoder_command(jp, jv)
        per_frame = np.concatenate([command, anchor_mf], axis=2)
        enc = per_frame.reshape(R, F * (_CMD_PER_FRAME * n + _ANCHOR_ORI6))

        bias = info.get("default_dof_pos_bias")
        effective_default = self.default_angles + bias if bias is not None else self.default_angles
        joint_pos_rel = np.asarray(dof_pos - effective_default, dtype=dtype)
        last_actions = info.get("current_actions")
        if not isinstance(last_actions, np.ndarray):
            last_actions = np.zeros((R, n), dtype=dtype)
        last_actions = np.asarray(last_actions, dtype=dtype)
        pelvis_quat = robot_body_quat_w[:, ai]
        gravity_dir = np_quat_apply_inverse(
            pelvis_quat, np.broadcast_to(_GRAVITY, (R, 3)).astype(pelvis_quat.dtype)
        ).astype(dtype)
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

        sel = slice(None) if env_ids is None else env_ids
        proprio_terms = [
            self._hist[k][sel].reshape(R, -1)
            for k in ("gyro", "joint_pos_rel", "dof_vel", "last_actions", "gravity_dir")
        ]
        proprio = np.concatenate(proprio_terms, axis=1)
        return np.concatenate([enc, proprio], axis=1, dtype=dtype)

    def _push_history(self, env_ids, components):
        sel = slice(None) if env_ids is None else env_ids
        for key, val in components.items():
            buf = self._hist[key]
            buf[sel, :-1] = buf[sel, 1:]
            buf[sel, -1] = val

    def _fill_history(self, env_ids, components):
        sel = slice(None) if env_ids is None else env_ids
        for key, val in components.items():
            self._hist[key][sel, :] = val[:, None, :]

    def _push_critic_history(self, env_ids, components):
        sel = slice(None) if env_ids is None else env_ids
        for key, val in components.items():
            buf = self._chist[key]
            buf[sel, :-1] = buf[sel, 1:]
            buf[sel, -1] = val

    def _fill_critic_history(self, env_ids, components):
        sel = slice(None) if env_ids is None else env_ids
        for key, val in components.items():
            self._chist[key][sel, :] = val[:, None, :]

    def _proprio_core(self, info, dof_pos, dof_vel, gyro):
        dtype = get_global_dtype()
        n = self._num_action
        R = dof_pos.shape[0]
        bias = info.get("default_dof_pos_bias")
        effective_default = self.default_angles + bias if bias is not None else self.default_angles
        joint_pos_rel = np.asarray(dof_pos - effective_default, dtype=dtype)
        last_actions = info.get("current_actions")
        if not isinstance(last_actions, np.ndarray):
            last_actions = np.zeros((R, n), dtype=dtype)
        last_actions = np.asarray(last_actions, dtype=dtype)
        dof_vel = np.asarray(dof_vel, dtype=dtype)
        return {
            "gyro": np.asarray(gyro, dtype=dtype),
            "joint_pos_rel": joint_pos_rel,
            "dof_vel": dof_vel,
            "last_actions": last_actions,
        }

    def _build_sonic_critic(
        self, info, motion_data, linvel, gyro, dof_pos, dof_vel,
        robot_body_pos_w, robot_body_quat_w,
    ) -> np.ndarray:
        """Byte-exact official privileged_mf_hist value observation (1745 for n=31)."""
        dtype = get_global_dtype()
        R = dof_pos.shape[0]
        ai = self.anchor_body_idx
        env_ids = info.get("env_ids")
        is_reset = env_ids is not None

        jp, jv, _, _ = self._gather_future(env_ids)
        command_mf = np.concatenate([jp.reshape(R, -1), jv.reshape(R, -1)], axis=1)

        anchor_pos = np.empty((R, 3), dtype=dtype)
        anchor_ori6 = np.empty((R, _ANCHOR_ORI6), dtype=dtype)
        np_write_relative_anchor_transform_pos_rot6d(
            robot_body_pos_w[:, ai], robot_body_quat_w[:, ai],
            motion_data.body_pos_w[:, ai], motion_data.body_quat_w[:, ai],
            anchor_pos, anchor_ori6,
        )

        bidx = self._critic_body_indices
        nb = bidx.size
        src_pos = np.repeat(robot_body_pos_w[:, ai], nb, axis=0)
        src_quat = np.repeat(robot_body_quat_w[:, ai], nb, axis=0)
        tgt_pos = robot_body_pos_w[:, bidx].reshape(R * nb, 3)
        tgt_quat = robot_body_quat_w[:, bidx].reshape(R * nb, 4)
        body_pos_out = np.empty((R * nb, 3), dtype=dtype)
        body_ori_out = np.empty((R * nb, _ANCHOR_ORI6), dtype=dtype)
        np_write_relative_anchor_transform_pos_rot6d(
            src_pos, src_quat, tgt_pos, tgt_quat, body_pos_out, body_ori_out
        )
        body_pos = body_pos_out.reshape(R, nb * 3)
        body_ori = body_ori_out.reshape(R, nb * _ANCHOR_ORI6)

        core = self._proprio_core(info, dof_pos, dof_vel, gyro)
        components = {
            "base_lin_vel": np.asarray(linvel, dtype=dtype),
            "base_ang_vel": core["gyro"],
            "joint_pos_rel": core["joint_pos_rel"],
            "joint_vel": core["dof_vel"],
            "last_actions": core["last_actions"],
        }
        if is_reset:
            self._fill_critic_history(env_ids, components)
        else:
            self._push_critic_history(env_ids, components)
        sel = slice(None) if env_ids is None else env_ids
        hist_terms = [
            self._chist[k][sel].reshape(R, -1)
            for k in ("base_lin_vel", "base_ang_vel", "joint_pos_rel", "joint_vel", "last_actions")
        ]
        return np.concatenate(
            [command_mf, anchor_pos, anchor_ori6, body_pos, body_ori, *hist_terms],
            axis=1, dtype=dtype,
        )

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        self._reward_fns["motion_local_points"] = self._reward_motion_local_points
        self._reward_fns["feet_slide"] = self._reward_feet_slide
        self._reward_fns["feet_yaw"] = self._reward_feet_yaw
        self._reward_fns["joint_acc_l2"] = self._reward_joint_acc_l2
        self._reward_fns["joint_torque_l2"] = self._reward_joint_torque_l2
        self._reward_fns["anti_shake_ang_vel"] = self._reward_anti_shake_ang_vel

    def _reward_joint_acc_l2(self, ctx: Any) -> np.ndarray:
        """Official ``feet_acc``: L2 of finite-difference ankle joint acceleration.

        ``acc = (dof_vel - prev_dof_vel) / ctrl_dt`` restricted to the ``.*ankle.*``
        joints. ``_prev_dof_vel`` is seeded on every reset in :meth:`_compute_obs`,
        so the first post-reset sample is a true Δv, not a spurious spike. Directly
        smooths the foot motion that drives feet_slide / feet_yaw and the global
        root-position drift seen in training.
        """
        idx = self._ankle_dof_indices
        if idx.size == 0:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        acc = (ctx.dof_vel[:, idx] - self._prev_dof_vel[:, idx]) / self._cfg.ctrl_dt
        return np.asarray(np.sum(np.square(acc), axis=1), dtype=get_global_dtype())

    def _reward_joint_torque_l2(self, ctx: Any) -> np.ndarray:
        """L2 of the PD torque applied this step: ``kp·(target_q − q) − kd·q̇``.

        ``target_q`` is the current residual action mapped through ``action_scale``
        onto the (bias-adjusted) default pose. Penalizes raw actuator effort — the
        principled smoothness term for H2, whose effort limits are ≈3× G1 so the
        same jitter costs far more torque. Stateless (no reset seeding needed).
        """
        if self._torque_kp is None or self._torque_kd is None:
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        actions = ctx.info.get("current_actions")
        if not isinstance(actions, np.ndarray):
            return np.zeros((self._num_envs,), dtype=get_global_dtype())
        base = self._effective_default_angles()
        target_q = actions * self._cfg.control_config.action_scale + base
        torque = self._torque_kp * (target_q - ctx.dof_pos) - self._torque_kd * ctx.dof_vel
        return np.asarray(np.sum(np.square(torque), axis=1), dtype=get_global_dtype())

    def _reward_anti_shake_ang_vel(self, ctx: Any) -> np.ndarray:
        """Deadzone jitter penalty on wrists (body ang-vel) and head (joint-vel).

        Wrists are tracked bodies → use world angular speed ``‖ω‖``. The head is not
        in the 14-body set and has no reference motion in the G1→H2 retarget, so its
        jitter is penalized in joint space via head dof velocity. Speeds below
        ``anti_shake_threshold`` incur no penalty, leaving intentional motion free.
        """
        thr = self._cfg.reward_config.anti_shake_threshold
        wv = ctx.robot_body_ang_vel_w[:, self._wrist_body_indices]
        wspeed = np.sqrt(np.sum(np.square(wv), axis=2))
        wexc = np.maximum(wspeed - thr, 0.0)
        penalty = np.mean(np.square(wexc), axis=1)
        if self._head_dof_indices.size:
            hexc = np.maximum(np.abs(ctx.dof_vel[:, self._head_dof_indices]) - thr, 0.0)
            penalty = penalty + np.mean(np.square(hexc), axis=1)
        return np.asarray(penalty, dtype=get_global_dtype())

    def _reward_feet_slide(self, ctx: Any) -> np.ndarray:
        """Penalize horizontal foot velocity while the foot is near the ground.

        Contact is proxied by robot foot world-z < ``feet_contact_height`` (flat
        ground; mirrors ``undesired_contacts``' height proxy, no contact sensor
        needed). Returns the per-foot-mean horizontal speed² of *contacting* feet;
        the negative ``feet_slide`` scale turns it into a slip penalty. Geometry-
        agnostic — shapes the policy, not the contact model, so train == deploy.
        """
        idx = self._strict_foot_indices
        np.square(ctx.robot_body_lin_vel_w[:, idx, :2], out=self._foot_xy_sq)
        np.sum(self._foot_xy_sq, axis=2, out=self._foot_perfoot)
        np.less(
            ctx.robot_body_pos_w[:, idx, 2],
            self._cfg.reward_config.feet_contact_height,
            out=self._foot_gate,
        )
        self._foot_perfoot *= self._foot_gate
        np.sum(self._foot_perfoot, axis=1, out=self._foot_penalty_slide)
        self._foot_penalty_slide /= idx.size
        return self._foot_penalty_slide

    def _reward_feet_yaw(self, ctx: Any) -> np.ndarray:
        """Penalize foot yaw-rate (spin about vertical) while near the ground.

        Same contact-height proxy as :meth:`_reward_feet_slide`. Directly targets
        the ``motion_body_ang_vel`` failure mode where H2's single-mesh foot, having
        no torsional friction, free-spins during single-support turns.
        """
        idx = self._strict_foot_indices
        np.square(ctx.robot_body_ang_vel_w[:, idx, 2], out=self._foot_perfoot)
        np.less(
            ctx.robot_body_pos_w[:, idx, 2],
            self._cfg.reward_config.feet_contact_height,
            out=self._foot_gate,
        )
        self._foot_perfoot *= self._foot_gate
        np.sum(self._foot_perfoot, axis=1, out=self._foot_penalty_yaw)
        self._foot_penalty_yaw /= idx.size
        return self._foot_penalty_yaw

    def _reward_motion_local_points(self, ctx: Any) -> np.ndarray:
        ref_pos = ctx.motion_data.body_pos_w[:, self._strict_local_point_indices]
        robot_pos = ctx.robot_body_pos_w[:, self._strict_local_point_indices]
        ref_anchor = ctx.motion_data.body_pos_w[:, self.anchor_body_idx]
        robot_anchor = ctx.robot_body_pos_w[:, self.anchor_body_idx]
        ref_quat = ctx.motion_data.body_quat_w[:, self.anchor_body_idx]
        robot_quat = ctx.robot_body_quat_w[:, self.anchor_body_idx]
        point_count = self._strict_local_point_indices.size
        ref_anchor_tiled = np.broadcast_to(ref_anchor[:, None, :], (self._num_envs, point_count, 3)).reshape(-1, 3)
        ref_quat_tiled = np.broadcast_to(ref_quat[:, None, :], (self._num_envs, point_count, 4)).reshape(-1, 4)
        robot_anchor_tiled = np.broadcast_to(robot_anchor[:, None, :], (self._num_envs, point_count, 3)).reshape(-1, 3)
        robot_quat_tiled = np.broadcast_to(robot_quat[:, None, :], (self._num_envs, point_count, 4)).reshape(-1, 4)
        np_write_relative_anchor_transform_pos_rot6d(
            ref_anchor_tiled, ref_quat_tiled, ref_pos.reshape(-1, 3), ref_quat_tiled,
            self._strict_point_reference.reshape(-1, 3), self._strict_point_rot6d,
        )
        np_write_relative_anchor_transform_pos_rot6d(
            robot_anchor_tiled, robot_quat_tiled, robot_pos.reshape(-1, 3), robot_quat_tiled,
            self._strict_point_error.reshape(-1, 3), self._strict_point_rot6d,
        )
        self._strict_point_error -= self._strict_point_reference
        np.square(self._strict_point_error, out=self._strict_point_error)
        np.sum(self._strict_point_error, axis=(1, 2), out=ctx.env_error)
        ctx.env_error /= self._strict_local_point_indices.size
        np.divide(ctx.env_error, -(self._cfg.reward_config.std_local_points**2), out=ctx.reward_term)
        np.exp(ctx.reward_term, out=ctx.reward_term)
        return ctx.reward_term

    def _compute_terminations(self, motion_data, robot_body_pos_w, robot_body_quat_w):
        terminated = super()._compute_terminations(motion_data, robot_body_pos_w, robot_body_quat_w)
        ref_anchor_pos = motion_data.body_pos_w[:, self.anchor_body_idx]
        robot_anchor_pos = robot_body_pos_w[:, self.anchor_body_idx]
        np.subtract(ref_anchor_pos[:, 2], robot_anchor_pos[:, 2], out=self._env_error)
        np.abs(self._env_error, out=self._env_error)
        np.greater(self._env_error, self._cfg.strict_height_threshold, out=self._env_bool)
        low_reference = ref_anchor_pos[:, 2] < self._cfg.low_reference_height
        np.greater(self._env_error, self._cfg.low_reference_height_threshold, out=self._strict_done)
        self._env_bool[low_reference] = self._strict_done[low_reference]
        terminated |= self._env_bool

        if self._has_ee_body_indices:
            np.subtract(
                self.body_pos_relative_w[:, self.ee_body_indices, 2],
                robot_body_pos_w[:, self.ee_body_indices, 2], out=self._ee_pos_error_z,
            )
            np.abs(self._ee_pos_error_z, out=self._ee_pos_error_z)
            np.greater(self._ee_pos_error_z, self._cfg.strict_height_threshold, out=self._ee_terminated)
            np.greater(self._ee_pos_error_z, self._cfg.low_reference_height_threshold, out=self._strict_ee_mask)
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
        self, info, motion_data, linvel, gyro, dof_pos, dof_vel,
        robot_body_pos_w, robot_body_quat_w,
    ) -> dict[str, np.ndarray]:
        base = super()._compute_obs(
            info, motion_data, linvel, gyro, dof_pos, dof_vel, robot_body_pos_w, robot_body_quat_w
        )
        actor = self._build_sonic_actor(info, dof_pos, dof_vel, gyro, robot_body_pos_w, robot_body_quat_w)

        # Cache dof_vel for next step's finite-difference joint_acc_l2. This runs on
        # every obs path: full step (env_ids None → all rows) and reset/clip-resample
        # (env_ids set → only those rows), so the acceleration never sees a reset jump.
        env_ids = info.get("env_ids")
        if env_ids is None:
            self._prev_dof_vel[:] = dof_vel
        else:
            self._prev_dof_vel[env_ids] = dof_vel
        if self._cfg.critic_privileged_mf_hist:
            critic = self._build_sonic_critic(
                info, motion_data, linvel, gyro, dof_pos, dof_vel, robot_body_pos_w, robot_body_quat_w,
            )
            return {"obs": actor, "critic": critic}
        critic = base["critic"]
        if self._cfg.critic_include_future:
            critic = np.concatenate([critic, actor[:, : self._enc_dim]], axis=1, dtype=critic.dtype)
        return {"obs": actor, "critic": critic}
