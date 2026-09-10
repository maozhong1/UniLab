"""MicroduckBaseEnv — the shared base for the whole microduck task family.

This is the *extensibility core* of the microduck→UniLab port. Every microduck
task (velocity/walk, standup, sitstand, ground-pick, roulade, rollers, swizzle,
and their backlash twins) subclasses ``MicroduckBaseEnv`` and only supplies its
reward dictionary, command usage, and termination extras. Everything that MUST
stay identical across the family so policies are hot-swappable in the deployed
runtime lives here, once:

* **61D actor obs contract** (sim2real-frozen, shared by the whole family):
  ``[base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14),
  last_action(14), twist(3), head_pose(4), body_pose(6)]`` = 48 proprio + 13
  command block, in exactly this order. The critic appends privileged
  ``base_lin_vel(3)`` → 64D. A task that doesn't use a command slot ZERO-PADS it
  (keeps the obs term, samples a tiny keep-alive range) — never deletes a slot,
  or the runtime obs layout breaks and every other policy stops loading.
* **projected_gravity is computed from the base quaternion** (``R^T·(0,0,-1)``),
  NOT read from a framezaxis/upvector sensor. microduck_rl's deployment +
  ``infer_policy.py`` feed ``projected_gravity_b``; the Phase-6 ONNX cross-check
  and the real robot both require this exact quantity, so we reproduce it rather
  than inherit g1's sensor convention.
* **BAM actuator** mounted as the backend pre-step control (see
  ``actuator.py``): the policy action is a HOME-relative joint-position target,
  and the BAM voltage law turns it into motor torque every physics substep.
* **servo joint indices are resolved by name**, never hardcoded — identity on
  the plain walk model, correct on the rollers/backlash models where passive
  joints interleave (invariant from AGENTS.md).

Subclasses implement ``_init_reward_functions`` (populate ``self._reward_fns``)
and set ``self._reward_cfg``; the base handles obs, termination, BAM wiring, DR,
and the reward dispatch. With no reward config the base still steps (reward=0),
so the obs/termination/actuator machinery is testable before any task exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)
from unilab.envs.locomotion.common.base import (
    Sensor as LocomotionSensor,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext, run_reward_dispatch
from unilab.envs.locomotion.microduck.actuator import BamActuatorConfig, MicroduckBamActuator
from unilab.utils.rotation import np_quat_apply_inverse

# ── the 14 servos, in the sim2real-frozen order (0-4 left leg, 5-8 neck/head,
#    9-13 right leg). ctrl column == servo index on the walk/groundcontact
#    models; the by-name index helpers keep it correct when passive joints
#    interleave on rollers/backlash. ──────────────────────────────────────────
SERVO_NAMES: tuple[str, ...] = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
NUM_SERVOS = len(SERVO_NAMES)

# Default leg-pose penalty weights (14-vector): leg joints = 1, head/neck (5-8)
# = 0. Head joints are driven by head_pose_tracking, so a pose penalty toward
# HOME on them would fight the command (AGENTS.md pose/head conflict lesson).
LEG_POSE_DEFAULT_WEIGHTS: tuple[float, ...] = (
    1.0, 1.0, 1.0, 1.0, 1.0,   # left leg
    0.0, 0.0, 0.0, 0.0,        # neck/head
    1.0, 1.0, 1.0, 1.0, 1.0,   # right leg
)

# Obs block sizes — the family-wide contract. Do not change without bumping the
# runtime obs schema and retraining every policy in the set.
NUM_TWIST = 3
NUM_HEAD_POSE = 4      # [neck_pitch, head_pitch, head_yaw, head_roll] target deltas from HOME
NUM_BODY_POSE = 6      # [x, y, z, roll, pitch, yaw] base-pose delta from nominal stand
NUM_COMMAND = NUM_TWIST + NUM_HEAD_POSE + NUM_BODY_POSE      # 13
NUM_PROPRIO = 3 + 3 + NUM_SERVOS + NUM_SERVOS + NUM_SERVOS  # 48
ACTOR_OBS_DIM = NUM_PROPRIO + NUM_COMMAND                   # 61
CRITIC_OBS_DIM = ACTOR_OBS_DIM + 3                          # 64 (+ base_lin_vel)

# World gravity direction (unit, pointing down) used to build projected_gravity_b.
_GRAVITY_DIR_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)

_MICRODUCK_ROOT = ASSETS_ROOT_PATH / "robots" / "microduck"


# ── configs ──────────────────────────────────────────────────────────────────
@dataclass
class MicroduckSensor(LocomotionSensor):
    # gyro in body frame (base_ang_vel obs); velocimeter = local linvel (critic).
    gyro: str = "imu_ang_vel"
    local_linvel: str = "imu_lin_vel"
    # whole-body angular momentum (subtreeangmom), reward-only (roller stride).
    root_angmom: str = "root_angmom"
    # Foot frame/contact sensors (reward-only, NOT in the 61D obs). Present on the
    # walk/groundcontact models; only tasks with a foot-gait reward read them (the
    # base env never does, so models without contact sensors still load).
    left_foot_pos: str = "left_foot_pos"
    right_foot_pos: str = "right_foot_pos"
    left_foot_vel: str = "left_foot_vel"
    right_foot_vel: str = "right_foot_vel"
    left_foot_contact: str = "left_foot_contact"
    right_foot_contact: str = "right_foot_contact"
    # Roller-only extras (present on the rollers model): foot-site orientation for
    # the feet_flat tilt penalty, and a whole-robot self-collision contact scalar.
    left_foot_quat: str = "left_foot_quat"
    right_foot_quat: str = "right_foot_quat"
    self_collision: str = "self_collision"


@dataclass
class MicroduckNoiseConfig(BaseNoiseConfig):
    """Actor-obs corruption ranges mirroring microduck_rl's velocity recipe.

    ``level`` scales all of them (uniform ``[-1,1]*level*scale``); training runs
    with ``level=1``, eval/ONNX export runs clean (``level=0``).
    """

    level: float = 1.0
    scale_gyro: float = 0.03         # base_ang_vel Unoise ±0.03
    scale_gravity: float = 0.01      # projected_gravity Unoise ±0.01
    scale_joint_angle: float = 0.001  # joint_pos Unoise ±0.001
    scale_joint_vel: float = 0.25    # joint_vel Unoise ±0.25
    scale_linvel: float = 0.0        # critic linvel is privileged/clean


@dataclass
class MicroduckControlConfig(ControlConfigBase):
    # ctrl = action*scale + HOME; microduck sets joint_pos action scale = 1.0 and
    # the deployed runtime applies the same, so this is deployment-matched.
    action_scale: float = 1.0
    simulate_action_latency: bool = False


@dataclass
class MicroduckCommandRanges:
    """Sampling ranges for the 13D command block.

    Defaults are the tiny *keep-alive* ranges: every slot samples a small
    non-zero range from step 0 so its input neurons never die, even when a task
    weights that slot at 0 (dead-weight prevention, AGENTS.md). Tasks widen the
    slots they actually drive (velocity → twist + head_pose; standup → body_pose).
    """

    # [[vx,vy,vyaw]_min, [...]_max]
    twist_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.1, -0.05, -0.1], [0.1, 0.05, 0.1]]
    )
    # per-joint (lo, hi): neck_pitch, head_pitch, head_yaw, head_roll
    head_pose_ranges: list[list[float]] = field(
        default_factory=lambda: [
            [-0.05, 0.05], [-0.05, 0.05], [-0.07, 0.07], [-0.015, 0.015],
        ]
    )
    # (lo, hi): x, y, z, roll, pitch, yaw
    body_pose_ranges: list[list[float]] = field(
        default_factory=lambda: [
            [-0.005, 0.005], [-0.005, 0.005], [-0.005, 0.005],
            [-0.05, 0.05], [-0.05, 0.05], [-0.05, 0.05],
        ]
    )
    # Fraction of resets given the exact all-zero command (idle / stand still) —
    # uniform sampling essentially never produces it, but it's the deploy state.
    zero_command_prob: float = 0.0
    # Fraction of resets forced to spin in place (twist xy=0, |vyaw| in the top band).
    turn_in_place_fraction: float = 0.0
    turn_in_place_min_frac: float = 0.4  # |vyaw| >= this * vyaw_max


@dataclass
class MicroduckDomainRandConfig(DomainRandConfig):
    """microduck DR. v1 randomizes BAM electricals (per-env, in the actuator) +
    per-reset BAM kp/kd + optional pushes/armature. Mass/CoM/geom-friction
    plumbing is inherited but off by default (widen in a task/YAML later)."""

    # Per-reset BAM firmware-gain scaling (electrical kp / back-EMF damping).
    randomize_bam_kp: bool = True
    bam_kp_scale_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    randomize_bam_kd: bool = False
    bam_kd_scale_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    push_body_name: str | None = "trunk_base"


@dataclass
class MicroduckBaseCfg(LocomotionBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(model_file=str(_MICRODUCK_ROOT / "scene_flat_motor.xml"))
    )
    sensor: MicroduckSensor = field(default_factory=MicroduckSensor)
    noise_config: MicroduckNoiseConfig = field(default_factory=MicroduckNoiseConfig)  # type: ignore[assignment]
    control_config: MicroduckControlConfig = field(default_factory=MicroduckControlConfig)  # type: ignore[assignment]
    commands: MicroduckCommandRanges = field(default_factory=MicroduckCommandRanges)
    domain_rand: MicroduckDomainRandConfig = field(default_factory=MicroduckDomainRandConfig)
    bam: BamActuatorConfig = field(default_factory=BamActuatorConfig)

    base_name: str = "trunk_base"
    # 50 Hz control over 200 Hz physics (substeps = 4), matching training.
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    max_episode_seconds: float | None = 20.0

    # Termination (fall) thresholds. min_base_height off the STAND trunk z (~0.12).
    max_tilt_deg: float = 60.0
    min_base_height: float = 0.06
    reset_base_qvel_limit: float = 0.5

    # Gait-phase clock init mode for foot-gait tasks (None = no clock; base tasks
    # unaffected). "offset_phase" = left/right seeded π apart (anti-phase stepping);
    # "independent" = each foot random. Set by tasks that use feet_phase.
    gait_phase_init_mode: str | None = None


# ── DR provider ──────────────────────────────────────────────────────────────
class MicroduckDRProvider(LocomotionDRProvider):
    """Reset/interval DR for microduck.

    Extends the shared locomotion provider with (a) 13D command-block sampling
    delegated to the env (so tasks control it) and (b) per-reset BAM firmware-gain
    randomization applied straight to the pre-step actuator via ``set_gains``.
    BAM electricals (vin, sag resistance) are per-env, sampled once at actuator
    construction and held across resets — non-accumulating by construction.
    """

    def __init__(self, *, base_dof_armature: np.ndarray | None = None):
        self._base_dof_armature = base_dof_armature
        self._rng = np.random.default_rng()

    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None, np.ndarray | None]:
        return None, None, None, self._base_dof_armature

    def _get_qvel_limit(self, env: Any) -> float:
        return float(env.cfg.reset_base_qvel_limit)

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        # Full 13D command block; the env owns the sampling policy so tasks can
        # override zero-command / turn-in-place behaviour.
        return env._sample_command_block(num_reset)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        # Per-reset BAM firmware-gain DR, applied directly to the pre-step
        # actuator (non-accumulating: re-sampled from cfg ranges each reset).
        act: MicroduckBamActuator = env._bam_actuator
        dr = env.cfg.domain_rand
        n = len(env_ids)
        kp = self._sample_scale(dr.randomize_bam_kp, dr.bam_kp_scale_range, n)
        kd = self._sample_scale(dr.randomize_bam_kd, dr.bam_kd_scale_range, n)
        act.set_gains(env_ids, kp_scale=kp, kd_scale=kd)
        act.reset(env_ids)  # zero the battery-sag torque memory
        return plan

    def _sample_scale(self, enabled: bool, rng_range: list[float], n: int) -> np.ndarray:
        if not enabled:
            return np.ones((n,), dtype=np.float64)
        lo, hi = float(rng_range[0]), float(rng_range[1])
        return self._rng.uniform(lo, hi, size=(n,))

    def _build_extra_info_updates(self, env: Any, num_reset: int) -> dict[str, np.ndarray]:
        # Seed the per-env gait-phase clock on reset for foot-gait tasks. Written
        # into info for the reset rows only (partial reset); apply_action advances
        # it every step. Off (empty) when no task opts in.
        mode = getattr(env.cfg, "gait_phase_init_mode", None)
        if not mode:
            return {}
        return {"gait_phase": self._sample_gait_phase(num_reset, mode)}

    def _sample_gait_phase(self, num_reset: int, mode: str) -> np.ndarray:
        """Per-env ``[left, right]`` gait phase in ``[0, 2π)``.

        ``offset_phase`` seeds the two feet π apart (anti-phase stepping);
        ``independent`` seeds each foot independently. Ported from g1's
        ``_sample_gait_phase``.
        """
        if mode == "independent":
            left = self._rng.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            right = self._rng.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            return np.column_stack([left, right]).astype(np.float32)
        phase = self._rng.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        return np.column_stack([phase, phase + np.pi]).astype(np.float32)

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        # microduck obs path: projected_gravity from the base quat, servo dofs by
        # name (not the base provider's upvector-sensor fetch).
        linvel = env.get_local_linvel()[env_ids]
        gyro = env.get_gyro()[env_ids]
        proj_grav = env._projected_gravity(env._backend.get_base_quat()[env_ids])
        dof_pos = env._servo_joint_pos()[env_ids]
        dof_vel = env._servo_joint_vel()[env_ids]
        return env._compute_obs(info_updates, linvel, gyro, proj_grav, dof_pos, dof_vel)


# ── env ──────────────────────────────────────────────────────────────────────
class MicroduckBaseEnv(LocomotionBaseEnv):
    """Shared base env for the microduck family. Subclass and add rewards."""

    _cfg: MicroduckBaseCfg
    _keyframe_name: ClassVar[str] = "STAND"  # STAND keyframe == HOME_FRAME joints
    _use_global_dtype: ClassVar[bool] = False  # float32 obs/action (ONNX-friendly)
    _reward_cfg: Any = None

    def __init__(self, cfg: MicroduckBaseCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            **env_backend_kwargs(cfg),
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True

        if self._num_action != NUM_SERVOS:
            raise ValueError(f"microduck expects {NUM_SERVOS} actuators, got {self._num_action}")

        # BAM actuator (voltage law) mounted as the backend pre-step control.
        self._bam_actuator = MicroduckBamActuator(cfg.bam, num_envs, list(SERVO_NAMES))

        # Subclass rewards (base default: none → reward 0, still steppable).
        self._reward_fns: dict[str, Any] = {}
        self._init_reward_functions()

        base_dof_armature = (
            backend.get_dof_armature() if cfg.domain_rand.randomize_dof_armature else None
        )
        self._init_domain_randomization(
            MicroduckDRProvider(base_dof_armature=base_dof_armature)
        )  # materializes the backend

        # Resolve servo dof indices from the compiled model (identity on walk;
        # correct where passive joints interleave) and arm the actuator.
        self._servo_pos_idx = np.asarray(
            self._backend.get_joint_dof_pos_indices(list(SERVO_NAMES))
        )
        self._servo_vel_idx = np.asarray(
            self._backend.get_joint_dof_vel_indices(list(SERVO_NAMES))
        )
        self._bam_actuator.bind(self._backend)
        self._backend.set_pre_step_control(self._bam_actuator)

    def _init_action_space(self) -> None:
        # The policy action is a normalized joint-position target (ctrl =
        # action*scale + HOME), NOT the motor force. The <motor> actuators are
        # ctrl-unlimited so their ctrlrange is a degenerate [0,0]; use a fixed
        # symmetric box (house pattern for motor-actuator envs, cf. go2w).
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_SERVOS,), dtype=np.float32
        )

    # ── obs contract ──────────────────────────────────────────────────────────
    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": ACTOR_OBS_DIM, "critic": CRITIC_OBS_DIM}

    def _projected_gravity(self, base_quat: np.ndarray) -> np.ndarray:
        """``projected_gravity_b`` = world down-vector expressed in the base frame.

        Matches microduck_rl's ``mdp.projected_gravity`` / the deployed obs, so
        the exported ONNX is directly runnable in ``infer_policy.py`` (Phase 6).
        """
        g = np.broadcast_to(_GRAVITY_DIR_W, (base_quat.shape[0], 3))
        return np.asarray(np_quat_apply_inverse(base_quat, g), dtype=np.float32)

    def _servo_joint_pos(self) -> np.ndarray:
        return np.asarray(self._backend.get_dof_pos())[:, self._servo_pos_idx]

    def _servo_joint_vel(self) -> np.ndarray:
        return np.asarray(self._backend.get_dof_vel())[:, self._servo_vel_idx]

    def _compute_obs(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        proj_grav: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        noise = self._cfg.noise_config
        diff = (dof_pos - self.default_angles).astype(np.float32)
        last_actions = info.get(
            "current_actions", np.zeros((diff.shape[0], NUM_SERVOS), dtype=np.float32)
        )
        command = info["commands"]  # (rows, 13)

        actor = np.concatenate(
            [
                self._obs_noise(gyro, noise.scale_gyro),
                self._obs_noise(proj_grav, noise.scale_gravity),
                self._obs_noise(diff, noise.scale_joint_angle),
                self._obs_noise(dof_vel, noise.scale_joint_vel),
                last_actions,
                command,
            ],
            axis=1,
            dtype=np.float32,
        )
        # Critic: same layout, clean, plus privileged base_lin_vel(3).
        critic = np.concatenate(
            [gyro, proj_grav, diff, dof_vel, last_actions, command, linvel],
            axis=1,
            dtype=np.float32,
        )
        return {"obs": actor, "critic": critic}

    def _obs_noise(self, data: np.ndarray, scale: float) -> np.ndarray:
        return np.asarray(super()._obs_noise(data, scale), dtype=data.dtype)

    # ── command block ──────────────────────────────────────────────────────────
    def _sample_command_block(self, num: int) -> np.ndarray:
        """Sample the 13D command block ``[twist(3), head(4), body(6)]``.

        Applies the keep-alive ranges plus the zero-command and turn-in-place
        buckets. Tasks override to change the policy (e.g. widen twist).
        """
        cmd = self._cfg.commands
        rng = np.random
        twist_lo = np.asarray(cmd.twist_limit[0], dtype=np.float32)
        twist_hi = np.asarray(cmd.twist_limit[1], dtype=np.float32)
        twist = rng.uniform(twist_lo, twist_hi, size=(num, NUM_TWIST)).astype(np.float32)

        if cmd.turn_in_place_fraction > 0.0:
            tip = rng.uniform(size=(num,)) < min(cmd.turn_in_place_fraction, 1.0)
            twist[tip, 0:2] = 0.0
            vyaw_max = float(max(abs(twist_lo[2]), abs(twist_hi[2])))
            band = rng.uniform(cmd.turn_in_place_min_frac * vyaw_max, vyaw_max, size=(num,))
            sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(num,))
            twist[tip, 2] = (band * sign).astype(np.float32)[tip]

        head = self._sample_ranges(cmd.head_pose_ranges, num)
        body = self._sample_ranges(cmd.body_pose_ranges, num)
        block = np.concatenate([twist, head, body], axis=1, dtype=np.float32)

        if cmd.zero_command_prob > 0.0:
            zero = rng.uniform(size=(num,)) < min(cmd.zero_command_prob, 1.0)
            block[zero] = 0.0  # exact idle: all slots → HOME target
        return block

    @staticmethod
    def _sample_ranges(ranges: list[list[float]], num: int) -> np.ndarray:
        lo = np.asarray([r[0] for r in ranges], dtype=np.float32)
        hi = np.asarray([r[1] for r in ranges], dtype=np.float32)
        return np.random.uniform(lo, hi, size=(num, lo.shape[0])).astype(np.float32)

    # ── step / termination / reward ─────────────────────────────────────────────
    def update_state(self, state: NpEnvState) -> NpEnvState:
        gyro = self.get_gyro()
        linvel = self.get_local_linvel()
        proj_grav = self._projected_gravity(self._backend.get_base_quat())
        dof_pos = self._servo_joint_pos()
        dof_vel = self._servo_joint_vel()

        # Fall termination: tilt (upright axis = -proj_grav_z) or sunk base.
        up = np.clip(-proj_grav[:, 2], -1.0, 1.0)
        tilt = np.arccos(up)
        base_z = self._backend.get_base_pos()[:, 2]
        terminated = (tilt > np.deg2rad(self._cfg.max_tilt_deg)) | (
            base_z < self._cfg.min_base_height
        )

        reward = self._compute_reward(state.info, linvel, gyro, proj_grav, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, proj_grav, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _build_reward_context(
        self, info: dict, linvel, gyro, proj_grav, dof_pos, dof_vel
    ) -> RewardContext:
        cfg = self._reward_cfg
        return RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            num_envs=dof_pos.shape[0],
            default_angles=self.default_angles,
            tracking_sigma=getattr(cfg, "tracking_sigma", 0.25),
            base_height_target=getattr(cfg, "base_height_target", 0.0),
            base_height=self._backend.get_base_pos()[:, 2],
            gravity=proj_grav,
            dof_vel=dof_vel,
        )

    def _compute_reward(self, info: dict, linvel, gyro, proj_grav, dof_pos, dof_vel) -> np.ndarray:
        cfg = self._reward_cfg
        if cfg is None or not self._reward_fns:
            return np.zeros((dof_pos.shape[0],), dtype=np.float32)
        ctx = self._build_reward_context(info, linvel, gyro, proj_grav, dof_pos, dof_vel)
        return run_reward_dispatch(
            scales=cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    # ── hooks for subclasses ─────────────────────────────────────────────────────
    def _init_reward_functions(self) -> None:
        """Populate ``self._reward_fns`` (name → callable). Base default: none."""
        self._reward_fns = {}

    # ── action → ctrl (position target consumed by the BAM pre-step) ───────────
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        scale = self._cfg.control_config.action_scale
        return (actions * scale + self.default_angles).astype(np.float32)
