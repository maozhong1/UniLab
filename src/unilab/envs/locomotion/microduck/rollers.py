"""MicroduckRollersEnv — roller-skate velocity (stride) task.

Port of microduck_rl ``microduck_velocity_rollers_env_cfg.py``: the duck skates
on 4 passive wheels (2 per blade). The ONLY positive task reward is
``wheel_speed`` (it must actually spin the wheels); ``braking`` /
``skating_air_time`` / ``glide`` / ``single_support`` / ``gait_symmetry`` /
``forward_lean`` / ``heading_hold`` shape a real alternating STRIDE (vs the
degenerate double-support swizzle).

Clock-FREE by construction: the gait terms read REAL per-foot air/contact times
(env timers), never a gait-phase clock (that path failed in the velocity port —
see docs/microduck_port_plan.md). Subclass of ``MicroduckBaseEnv``: inherits the
61D obs contract / BAM actuator / DR / termination; adds wheel + foot + torque
reads and the stride reward dictionary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.common import rewards as common_rewards
from unilab.envs.locomotion.microduck import rewards as md_rewards
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.microduck.base import (
    MicroduckBaseCfg,
    MicroduckBaseEnv,
    MicroduckCommandRanges,
    SERVO_NAMES,
    _MICRODUCK_ROOT,
)
from unilab.envs.locomotion.microduck.rewards import LEG_SERVO_IDX

WHEEL_NAMES = ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"]
HIP_ROLL_IDX = np.array([1, 10])  # hip_roll L/R in the 14-servo order


# ── per-servo std vectors from the mjlab regex dicts (rollers std_* tables) ─────
def _std_vec(std_dict: dict[str, float]) -> np.ndarray:
    """Resolve a ``{regex: std}`` dict to a (14,) per-servo std vector."""
    out = np.ones(len(SERVO_NAMES), dtype=np.float32)
    for i, name in enumerate(SERVO_NAMES):
        for pat, val in std_dict.items():
            if re.search(pat, name):
                out[i] = float(val)
                break
    return out


# passive_* std is irrelevant here (we only score the 14 servos); the leg/neck/head
# entries below match microduck_rl's roller std tables 1:1.
_STD_STANDING = {r".*hip_yaw.*": 0.05, r".*hip_roll.*": 0.05, r".*hip_pitch.*": 0.05,
                 r".*knee.*": 0.05, r".*ankle.*": 0.05, r".*neck.*": 0.05, r".*head.*": 0.05}
_STD_WALKING = {r".*hip_yaw.*": 0.3, r".*hip_roll.*": 0.6, r".*hip_pitch.*": 0.4,
                r".*knee.*": 0.4, r".*ankle.*": 0.25, r".*neck.*": 0.05, r".*head.*": 0.05}
_STD_RUNNING = {r".*hip_yaw.*": 0.5, r".*hip_roll.*": 0.8, r".*hip_pitch.*": 0.8,
                r".*knee.*": 0.8, r".*ankle.*": 0.5, r".*neck.*": 0.05, r".*head.*": 0.05}


def _rollers_command_ranges() -> MicroduckCommandRanges:
    """Stride command: cmd_x in (-0.5, 0.6) (0=coast, >0=push, <0=brake); no yaw
    (heading held via reward, not commanded); head/body slots ZERO (roller policy
    deploys with them zero-padded)."""
    return MicroduckCommandRanges(
        twist_limit=[[-0.5, 0.0, 0.0], [0.6, 0.0, 0.0]],
        head_pose_ranges=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        body_pose_ranges=[[0.0, 0.0]] * 6,
    )


@dataclass
class MicroduckRollersRewardCfg:
    """Stride reward config (YAML ``reward:`` block). ``scales`` = term→weight;
    the rest are the source kernel constants (overridable in YAML)."""

    scales: dict[str, float]
    upright_std: float = 0.2236
    # com height band (m)
    com_height_min: float = 0.0935
    com_height_max: float = 0.1235
    # wheel_speed
    wheel_vel_scale: float = 0.3
    wheel_radius: float = 0.0175
    wheel_bidirectional: bool = False
    # braking / gates
    braking_vel_std: float = 0.3
    # skating_air_time
    air_threshold_min: float = 0.15
    air_threshold_max: float = 0.45
    air_vel_gate_ref: float = 0.2
    # single_support
    single_vel_gate_ref: float = 0.2
    single_double_penalty: float = 0.25
    # glide
    glide_vel_ref: float = 0.2
    glide_stillness_std: float = 5.0
    # forward_lean
    forward_lean_target_pitch: float = 0.262
    forward_lean_std: float = 0.1
    # heading_hold
    heading_hold_std: float = 0.4
    # action_over_limit
    action_overshoot: float = 0.3
    # pose regime thresholds
    pose_walking_threshold: float = 0.01
    pose_running_threshold: float = 0.5
    # foot contact threshold (sensor scalar → in-contact)
    foot_contact_threshold: float = 0.5
    # ── action_rate curriculum (source: -1.0 → -1.5 @250it → -2.0 @500it) ────────
    # Calmer gait lever, ramped in AFTER the stride exists. wheel_friction / com DR
    # curricula are deferred (need the passive-wheel-frictionloss + CoM DR plumbing;
    # AGENTS: introduce DR after skill discovery — see port plan S5).
    curriculum_enabled: bool = True
    curriculum_num_steps_per_env: int = 24
    action_rate_stage2_iter: int = 250
    action_rate_stage3_iter: int = 500
    action_rate_stage2_weight: float = -1.5
    action_rate_stage3_weight: float = -2.0


@registry.envcfg("MicroduckRollers")
@dataclass
class MicroduckRollersCfg(MicroduckBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(model_file=str(_MICRODUCK_ROOT / "scene_rollers_motor.xml"))
    )
    commands: MicroduckCommandRanges = field(default_factory=_rollers_command_ranges)
    reward_config: MicroduckRollersRewardCfg | None = None
    # gait_phase_init_mode stays None (clock-free).


class MicroduckRollersEnv(MicroduckBaseEnv):
    _cfg: MicroduckRollersCfg

    def __init__(self, cfg: MicroduckRollersCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("MicroduckRollersEnv requires reward_config (owner YAML reward: block)")
        self._reward_cfg = cfg.reward_config
        # Pose std vectors (14,) resolved once.
        self._std_standing = _std_vec(_STD_STANDING)
        self._std_walking = _std_vec(_STD_WALKING)
        self._std_running = _std_vec(_STD_RUNNING)
        # Per-foot air/contact timers + cumulative swing accumulator + spawn yaw.
        self._current_air_time = np.zeros((num_envs, 2), dtype=np.float32)
        self._current_contact_time = np.zeros((num_envs, 2), dtype=np.float32)
        self._swing_accum = np.zeros((num_envs, 2), dtype=np.float32)
        self._heading_ref = np.zeros((num_envs,), dtype=np.float32)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        # Passive wheel dof-velocity indices (by name; interleaved on this model).
        self._wheel_vel_idx = np.asarray(self._backend.get_joint_dof_vel_indices(WHEEL_NAMES))
        # Servo hard limits (14,2) for action_over_limit.
        jr = self._backend.get_joint_range()
        self._servo_joint_range = np.asarray(jr)[self._servo_vel_idx] if jr is not None else None

    # ── reward wiring ────────────────────────────────────────────────────────────
    def _init_reward_functions(self) -> None:
        rc = self._reward_cfg
        self._reward_fns = {
            # positive-weight task/shaping rewards
            "wheel_speed": lambda ctx: md_rewards.wheel_speed_reward(
                ctx, self._wheel_forward_omega(), rc.wheel_vel_scale, rc.wheel_radius, rc.wheel_bidirectional),
            "braking": lambda ctx: md_rewards.braking_reward(ctx, rc.braking_vel_std),
            "skating_air_time": lambda ctx: md_rewards.skating_air_time_reward(
                ctx, self._current_air_time, rc.air_threshold_min, rc.air_threshold_max, rc.air_vel_gate_ref),
            "glide": lambda ctx: md_rewards.glide_reward(
                ctx, self._foot_contact(), self._leg_joint_vel_sq(), rc.glide_vel_ref, rc.glide_stillness_std),
            "single_support": lambda ctx: md_rewards.single_support_reward(
                ctx, self._foot_contact(), rc.single_vel_gate_ref, rc.single_double_penalty),
            "forward_lean": lambda ctx: md_rewards.forward_lean_reward(
                ctx, rc.forward_lean_target_pitch, rc.forward_lean_std),
            "heading_hold": lambda ctx: md_rewards.heading_hold_reward(ctx, self._heading_err(), rc.heading_hold_std),
            "com_height_target": lambda ctx: md_rewards.com_height_target(ctx, rc.com_height_min, rc.com_height_max),
            "upright": lambda ctx: md_rewards.upright(ctx, rc.upright_std),
            "pose": lambda ctx: md_rewards.variable_posture(
                ctx, self._std_standing, self._std_walking, self._std_running,
                rc.pose_walking_threshold, rc.pose_running_threshold),
            # negative-weight penalties (≥0)
            "body_ang_vel": common_rewards.ang_vel_xy,
            "angular_momentum": lambda ctx: md_rewards.angular_momentum_penalty(ctx, self._angmom()),
            "action_rate": common_rewards.action_rate,
            "self_collisions": lambda ctx: self._self_collision_count(),
            "feet_flat": lambda ctx: md_rewards.feet_flat_penalty(ctx, self._foot_tilt_xy(), self._foot_contact()),
            "neck_action_rate": md_rewards.neck_action_rate_l2,
            "joint_torques": common_rewards.dof_torques_l2,
            "action_over_limit": lambda ctx: self._action_over_limit(ctx),
            "hip_roll_neutral": lambda ctx: common_rewards.joint_deviation_l1(ctx, HIP_ROLL_IDX),
            "gait_symmetry": lambda ctx: md_rewards.gait_symmetry_penalty(ctx, self._swing_accum),
        }

    # ── sensor / state reads ─────────────────────────────────────────────────────
    def _wheel_forward_omega(self) -> np.ndarray:
        vel = np.asarray(self._backend.get_dof_vel())[:, self._wheel_vel_idx]  # (N,4)
        return vel.mean(axis=1).astype(np.float32)

    def _foot_contact(self) -> np.ndarray:
        thr = self._reward_cfg.foot_contact_threshold
        lc = self._backend.get_sensor_data(self._cfg.sensor.left_foot_contact)[:, 0]
        rc = self._backend.get_sensor_data(self._cfg.sensor.right_foot_contact)[:, 0]
        return np.column_stack([lc > thr, rc > thr])

    def _self_collision_count(self) -> np.ndarray:
        return self._backend.get_sensor_data(self._cfg.sensor.self_collision)[:, 0].astype(np.float32)

    def _leg_joint_vel_sq(self) -> np.ndarray:
        v = self._servo_joint_vel()[:, LEG_SERVO_IDX]
        return np.sum(np.square(v), axis=1).astype(np.float32)

    def _angmom(self) -> np.ndarray:
        return self._backend.get_sensor_data(self._cfg.sensor.root_angmom).astype(np.float32)

    def _foot_tilt_xy(self) -> np.ndarray:
        """Per-foot xy² of world-down projected into the foot-site frame (feet_flat)."""
        def tilt(name: str) -> np.ndarray:
            q = self._backend.get_sensor_data(name)  # (N,4) [w,x,y,z]
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            # gravity (0,0,-1) in site frame = -(third row of R); xy components:
            px = -(2.0 * (x * z - w * y))
            py = -(2.0 * (y * z + w * x))
            return px * px + py * py
        return np.column_stack([tilt(self._cfg.sensor.left_foot_quat),
                                tilt(self._cfg.sensor.right_foot_quat)]).astype(np.float32)

    def _base_yaw(self) -> np.ndarray:
        q = self._backend.get_base_quat()  # (N,4) [w,x,y,z]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)

    def _heading_err(self) -> np.ndarray:
        err = self._base_yaw() - self._heading_ref
        return np.arctan2(np.sin(err), np.cos(err)).astype(np.float32)

    def _action_over_limit(self, ctx) -> np.ndarray:
        if self._servo_joint_range is None:
            return np.zeros((ctx.num_envs,), dtype=np.float32)
        actions = ctx.info["current_actions"]
        target = actions * self._cfg.control_config.action_scale + self.default_angles
        return md_rewards.action_over_limit_penalty(
            ctx, target, self._servo_joint_range, self._reward_cfg.action_overshoot)

    # ── action_rate curriculum (step function on step_counter) ───────────────────
    def _maybe_apply_curriculum(self) -> None:
        rc = self._reward_cfg
        if not rc.curriculum_enabled or "action_rate" not in rc.scales:
            return
        it = self.step_counter // rc.curriculum_num_steps_per_env
        if it >= rc.action_rate_stage3_iter:
            rc.scales["action_rate"] = rc.action_rate_stage3_weight
        elif it >= rc.action_rate_stage2_iter:
            rc.scales["action_rate"] = rc.action_rate_stage2_weight

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        self._maybe_apply_curriculum()
        return super().apply_action(actions, state)

    # ── timers (clock-free gait) + info stamping ─────────────────────────────────
    def _update_contact_timers(self) -> None:
        contact = self._foot_contact()
        dt = self._cfg.ctrl_dt
        self._current_air_time[contact] = 0.0
        self._current_air_time[~contact] += dt
        self._current_contact_time[~contact] = 0.0
        self._current_contact_time[contact] += dt
        # cumulative swing time per foot (for gait_symmetry); reset in reset().
        self._swing_accum += (~contact).astype(np.float32) * dt

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_contact_timers()
        # BAM torque for joint_torques (dof_torques_l2 reads info["torques"]).
        state.info["torques"] = np.asarray(self._bam_actuator._prev_torque, dtype=np.float32)
        return super().update_state(state)

    def reset(self, env_indices: np.ndarray):
        obs, info = super().reset(env_indices)
        ids = np.asarray(env_indices, dtype=np.intp)
        if self._current_air_time.shape[0] == self._num_envs:
            self._current_air_time[ids] = 0.0
            self._current_contact_time[ids] = 0.0
            self._swing_accum[ids] = 0.0
            self._heading_ref[ids] = self._base_yaw()[ids]  # capture spawn heading
        return obs, info


registry.register_env("MicroduckRollers", MicroduckRollersEnv, sim_backend="mujoco")
