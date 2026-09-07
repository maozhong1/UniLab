"""MicroduckVelocityEnv — velocity-command walking (the first trainable task).

Thin subclass of ``MicroduckBaseEnv``: it only supplies the reward dictionary,
the widened command ranges (twist + a modest head_pose objective), and inherits
the entire 61D obs contract / BAM actuator / DR / termination stack.

The base reward is **IMU-only**: velocity tracking + head-pose tracking +
uprightness + alive, regularized by leg-pose / action-rate / vertical-bounce /
roll-pitch-rate penalties. Phase 4.5 adds **foot-gait** terms on top (they need
the foot sensors now in the scene MJCF, but do NOT enter the 61D obs):
``feet_phase`` (gait-clock swing-height tracking — forces real foot lift) and
``feet_slide`` (anti-skate penalty on horizontal foot speed while in contact).
See ``docs/microduck_port_plan.md`` Phase 4.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.common import rewards as common_rewards
from unilab.envs.locomotion.microduck import rewards as md_rewards
from unilab.envs.locomotion.microduck.base import (
    LEG_POSE_DEFAULT_WEIGHTS,
    MicroduckBaseCfg,
    MicroduckBaseEnv,
    MicroduckCommandRanges,
)


def _velocity_command_ranges() -> MicroduckCommandRanges:
    """Walking command distribution: real twist range + a modest head objective.

    body_pose stays at the tiny keep-alive range (slot carried for obs parity,
    weight 0). zero-command + turn-in-place buckets are explicitly trained.
    """
    return MicroduckCommandRanges(
        twist_limit=[[-0.4, -0.3, -1.0], [0.8, 0.3, 1.0]],
        head_pose_ranges=[
            [-0.2, 0.2],    # neck_pitch
            [-0.2, 0.2],    # head_pitch
            [-0.3, 0.3],    # head_yaw
            [-0.05, 0.05],  # head_roll
        ],
        # body_pose keep-alive (defaults on MicroduckCommandRanges) left as-is.
        zero_command_prob=0.1,
        turn_in_place_fraction=0.15,
    )


@dataclass
class MicroduckVelocityRewardCfg:
    """Reward config populated from the owner YAML ``reward:`` block.

    ``scales`` maps reward-term name → weight (UniLab sign convention: rewards
    ≥0 with positive weight, penalties ≥0 with negative weight). The remaining
    fields are the tracking/pose kernel widths. Required fields (no defaults) so
    the Hydra override constructs it via ``cls(**reward_dict)``.
    """

    scales: dict[str, float]
    lin_vel_sigma: float = 0.1
    ang_vel_sigma: float = 0.5
    head_pose_std: float = 0.5
    upright_std: float = 0.2236  # sqrt(0.05), matching microduck's upright kernel
    pose_weights: list[float] | None = None
    # Foot-gait (Phase 4.5) kernel params. Defaults are microduck-scale
    # placeholders; the owner YAML sets the measured values (see the measurement
    # step in the port plan). swing_height/ground_z in metres.
    gait_frequency: float = 1.5            # Hz; gait clock speed
    gait_duty: float = 0.6                 # stance fraction of the cycle (walk ≈ 0.6)
    feet_swing_height: float = 0.02        # target peak foot-site clearance
    feet_ground_z: float = 0.0             # measured stance foot-site world Z
    feet_phase_sigma: float = 4.0e-4       # exp kernel width (scaled to swing_height)
    feet_slide_contact_threshold: float = 0.5  # contact-sensor scalar → in-contact
    feet_min_cmd_norm: float = 0.05        # twist-norm gate for feet_phase
    feet_clearance_target: float = 0.03    # clock-free swing apex height (feet_clearance)
    # Air-time LANDING reward (feet_air_time_landing). Settles once per step at
    # touchdown: pays clip(last_air_time − threshold, ±cap) per landing foot —
    # SIGNED, so short swings are penalised and long swings rewarded, with a
    # smooth gradient both sides of the threshold (fixes the hard-floor cliff that
    # left Run B/C stuck at ~0 air-time; see port plan). Replaces the windowed
    # reward. threshold ≈ target minimum swing (s); cap bounds the per-landing pay.
    feet_air_time_landing_threshold: float = 0.20
    feet_air_time_landing_cap: float = 0.15

    # ── 2-stage gait curriculum ──────────────────────────────────────────────────
    # The base fields above (+ YAML scales) are STAGE A ("break the drag": strong
    # double_stance + air_time, fast clock → get any alternating stepping). At
    # ``curriculum_stage_b_iter`` the env switches to STAGE B ("refine to slow,
    # HIGH steps"): relax double_stance, slow the clock, and crank the height
    # (feet_phase/swing_height). A single static weight set can't serve both — the
    # break and refine phases want opposite weights (see port plan analysis).
    curriculum_enabled: bool = True
    curriculum_stage_b_iter: int = 150
    curriculum_num_steps_per_env: int = 24  # env steps per PPO iter (iter×this = step)
    # Stage-B overrides (applied once at the boundary):
    # double_stance MUST stay negative — a 1000-iter run with it at 0.0 collapsed
    # feet_air_time to ~0 for 185 iters (policy relaxed into a double-stance tap
    # shuffle: no pressure off double support + the windowed air-time's 0.15s hard
    # floor gives no gradient below it). Restore -0.5: it supplies the low-end
    # "leave double support" gradient while the windowed air-time supplies the
    # high-end "make swings LONG (slow)" reward. Unlike the old air-time, the
    # windowed one pays 0 for fast steps, so -0.5 no longer degenerates into a run.
    stage_b_feet_double_stance: float = -0.5
    stage_b_feet_phase: float = 3.0            # reward lifting HIGH
    stage_b_feet_phase_contrast: float = 3.0   # track the (now slower) clock harder
    stage_b_feet_contact_schedule: float = 3.0  # tie CONTACT (cadence) to the clock, hard
    stage_b_feet_air_time: float = 10.0        # longer swings → more air time
    stage_b_feet_slide: float = -0.2           # trim residual slip
    stage_b_feet_swing_height: float = 0.03    # aim higher
    stage_b_gait_frequency: float = 1.0        # slower cadence


@registry.envcfg("MicroduckVelocityFlat")
@dataclass
class MicroduckVelocityFlatCfg(MicroduckBaseCfg):
    commands: MicroduckCommandRanges = field(default_factory=_velocity_command_ranges)
    reward_config: MicroduckVelocityRewardCfg | None = None
    # Anti-phase gait clock for the feet_phase reward (seeded π apart per reset).
    gait_phase_init_mode: str | None = "offset_phase"


class MicroduckVelocityEnv(MicroduckBaseEnv):
    _cfg: MicroduckVelocityFlatCfg

    def __init__(self, cfg: MicroduckVelocityFlatCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("MicroduckVelocityEnv requires reward_config (owner YAML reward: block)")
        # Set before super().__init__ — the base calls _init_reward_functions().
        self._reward_cfg = cfg.reward_config
        pw = cfg.reward_config.pose_weights
        self._leg_pose_weights = np.asarray(
            pw if pw is not None else LEG_POSE_DEFAULT_WEIGHTS, dtype=np.float32
        )
        # Per-foot contact timers for the air-time reward. Init BEFORE super() —
        # backend materialization can trigger a reset (→ our reset() override).
        self._current_air_time = np.zeros((num_envs, 2), dtype=np.float32)
        self._current_contact_time = np.zeros((num_envs, 2), dtype=np.float32)
        # Landing bookkeeping for feet_air_time_landing: the air time captured at
        # the instant a foot touches down, and which feet landed this step.
        self._last_air_time = np.zeros((num_envs, 2), dtype=np.float32)
        self._first_contact = np.zeros((num_envs, 2), dtype=bool)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        # Gait-clock advance per control step (rad); paired with the anti-phase
        # reset seeding in MicroduckDRProvider and consumed by feet_phase(_contrast).
        # Mutable: the Stage-B curriculum recomputes it from a slower frequency.
        self._gait_phase_delta = float(
            2.0 * math.pi * cfg.reward_config.gait_frequency * cfg.ctrl_dt
        )
        # 2-stage gait curriculum bookkeeping.
        rc = cfg.reward_config
        self._stage_b_applied = False
        self._stage_b_start_step = int(
            rc.curriculum_stage_b_iter * rc.curriculum_num_steps_per_env
        )

    def _init_reward_functions(self) -> None:
        rc = self._reward_cfg
        w = self._leg_pose_weights
        self._reward_fns = {
            # positive-weight rewards
            "track_linear_velocity": lambda ctx: md_rewards.tracking_lin_vel(ctx, rc.lin_vel_sigma),
            "track_angular_velocity": lambda ctx: md_rewards.tracking_ang_vel(ctx, rc.ang_vel_sigma),
            "head_pose_tracking": lambda ctx: md_rewards.head_pose_tracking(ctx, rc.head_pose_std),
            "upright": lambda ctx: md_rewards.upright(ctx, rc.upright_std),
            "alive": common_rewards.alive,
            # foot-gait (all positive weight, ≥0):
            # (1) swing-foot height tracks the gait clock,
            "feet_phase": lambda ctx: md_rewards.feet_phase(
                ctx,
                self._foot_pos_z(),
                rc.feet_swing_height,
                rc.feet_ground_z,
                rc.feet_phase_sigma,
                rc.feet_min_cmd_norm,
                rc.gait_duty,
            ),
            # (2) L/R height DIFFERENCE tracks the clock — breaks the hover basin,
            "feet_phase_contrast": lambda ctx: md_rewards.feet_phase_contrast(
                ctx,
                self._foot_pos_z(),
                rc.feet_swing_height,
                rc.feet_phase_sigma,
                rc.feet_min_cmd_norm,
                rc.gait_duty,
            ),
            # (2b) CONTACT state tracks the clock schedule — the cadence lever. Ties
            #      foot planted/airborne to the duty-cycle clock; a fast shuffle out
            #      of phase with the slow clock scores low → forces clock-synced cadence.
            "feet_contact_schedule": lambda ctx: md_rewards.feet_contact_schedule(
                ctx, self._foot_contact(), rc.gait_duty, rc.feet_min_cmd_norm
            ),
            # (3) LANDING air time — the dominant driver, settled at touchdown.
            #     SIGNED: a swing shorter than the threshold lands NEGATIVE (fast
            #     shuffle is penalised), a longer swing lands POSITIVE, with a
            #     smooth gradient both sides → pulls swings LONGER (slower cadence)
            #     from any starting point (microduck_rl parity). Reads the landing
            #     bookkeeping the env stamps each step. Positive weight; signed value.
            "feet_air_time": lambda ctx: md_rewards.feet_air_time_landing(
                ctx,
                self._last_air_time,
                self._first_contact,
                rc.feet_air_time_landing_threshold,
                rc.feet_air_time_landing_cap,
                rc.feet_min_cmd_norm,
            ),
            # negative-weight penalties (return ≥0)
            "leg_pose": lambda ctx: md_rewards.leg_pose_l2(ctx, w),
            "action_rate": common_rewards.action_rate,
            "lin_vel_z": common_rewards.lin_vel_z,
            "ang_vel_xy": common_rewards.ang_vel_xy,
            # anti-skate: penalize planted-foot horizontal speed (negative weight).
            "feet_slide": lambda ctx: md_rewards.feet_slide(
                ctx, self._foot_speed_xy(), self._foot_contact()
            ),
            # clock-free swing clearance cost: pull a horizontally-moving foot to the
            # apex height `feet_clearance_target` (sets step HEIGHT; negative weight).
            "feet_clearance": lambda ctx: md_rewards.feet_clearance(
                ctx,
                self._foot_pos_z(),
                self._foot_speed_xy(),
                rc.feet_ground_z,
                rc.feet_clearance_target,
            ),
            # anti-drag: penalize both-feet-on-ground while moving — pushes the
            # policy out of the double-stance shuffle basin (negative weight).
            "feet_double_stance": lambda ctx: md_rewards.feet_double_stance(
                ctx, self._foot_contact(), rc.feet_min_cmd_norm
            ),
        }

    # ── foot-sensor reads (reward-only; not in obs) ──────────────────────────────
    def _foot_pos_z(self) -> np.ndarray:
        """World Z of the [left, right] foot sites, shape ``(N, 2)``."""
        left = self._backend.get_sensor_data(self._cfg.sensor.left_foot_pos)[:, 2]
        right = self._backend.get_sensor_data(self._cfg.sensor.right_foot_pos)[:, 2]
        return np.column_stack([left, right]).astype(np.float32)

    def _foot_speed_xy(self) -> np.ndarray:
        """Horizontal world speed of each foot, shape ``(N, 2)`` (left, right)."""
        lv = self._backend.get_sensor_data(self._cfg.sensor.left_foot_vel)
        rv = self._backend.get_sensor_data(self._cfg.sensor.right_foot_vel)
        left = np.linalg.norm(lv[:, :2], axis=1)
        right = np.linalg.norm(rv[:, :2], axis=1)
        return np.column_stack([left, right]).astype(np.float32)

    def _foot_contact(self) -> np.ndarray:
        """Boolean ground contact per foot, shape ``(N, 2)`` (left, right)."""
        thr = self._reward_cfg.feet_slide_contact_threshold
        lc = self._backend.get_sensor_data(self._cfg.sensor.left_foot_contact)[:, 0]
        rc = self._backend.get_sensor_data(self._cfg.sensor.right_foot_contact)[:, 0]
        return np.column_stack([lc > thr, rc > thr])

    # ── 2-stage gait curriculum ──────────────────────────────────────────────────
    def _maybe_apply_curriculum(self) -> None:
        """At the Stage-A→B boundary, switch to the 'refine' weights (once).

        Stage A breaks the drag basin; Stage B (relaxed double_stance + slower
        clock + stronger height reward) converts the fast/low penguin shuffle into
        slow, high steps. Weights and swing_height are read from cfg every step, so
        mutating them in place takes effect immediately; the gait clock step is
        recomputed from the slower Stage-B frequency.
        """
        rc = self._reward_cfg
        if self._stage_b_applied or not rc.curriculum_enabled:
            return
        if self.step_counter < self._stage_b_start_step:
            return
        sc = rc.scales
        stage_b = {
            "feet_double_stance": rc.stage_b_feet_double_stance,
            "feet_phase": rc.stage_b_feet_phase,
            "feet_phase_contrast": rc.stage_b_feet_phase_contrast,
            "feet_contact_schedule": rc.stage_b_feet_contact_schedule,
            "feet_air_time": rc.stage_b_feet_air_time,
            "feet_slide": rc.stage_b_feet_slide,
        }
        for name, weight in stage_b.items():
            if name in sc:  # don't activate a term the owner YAML left out
                sc[name] = weight
        rc.feet_swing_height = rc.stage_b_feet_swing_height
        self._gait_phase_delta = float(
            2.0 * math.pi * rc.stage_b_gait_frequency * self._cfg.ctrl_dt
        )
        self._stage_b_applied = True
        print(
            f"[microduck curriculum] Stage B applied at step {self.step_counter} "
            f"(iter ~{self.step_counter // rc.curriculum_num_steps_per_env}): "
            f"double_stance={rc.stage_b_feet_double_stance}, feet_phase={rc.stage_b_feet_phase}, "
            f"air_time={rc.stage_b_feet_air_time}, swing_height={rc.stage_b_feet_swing_height}, "
            f"gait_freq={rc.stage_b_gait_frequency}",
            flush=True,
        )

    # ── action → ctrl, advancing the gait clock first ────────────────────────────
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        self._maybe_apply_curriculum()
        gait_phase = state.info.get(
            "gait_phase", np.zeros((actions.shape[0], 2), dtype=np.float32)
        )
        gait_phase = (gait_phase + self._gait_phase_delta) % (2.0 * np.pi)
        state.info["gait_phase"] = gait_phase.astype(np.float32)
        return super().apply_action(actions, state)

    # ── contact timers for the air-time reward ───────────────────────────────────
    def _update_contact_timers(self) -> None:
        """Advance per-foot air/contact timers from the current contact state.

        ``current_air_time`` counts time since a foot left the ground;
        ``current_contact_time`` time since it landed. Consumed by
        ``feet_air_time_positive_biped`` (via ``info``). Ported from go2's
        ``_update_contact_timers``.
        """
        contact = self._foot_contact()  # (N, 2) bool
        dt = self._cfg.ctrl_dt
        # Landing detection BEFORE resetting the air timer: a foot "just landed"
        # if it is in contact now and had accumulated air time last step. Stash the
        # air time at that instant for the landing reward (feet_air_time_landing).
        self._first_contact = contact & (self._current_air_time > 0.0)
        self._last_air_time = self._current_air_time.copy()
        self._current_air_time[contact] = 0.0
        self._current_air_time[~contact] += dt
        self._current_contact_time[~contact] = 0.0
        self._current_contact_time[contact] += dt

    def update_state(self, state: NpEnvState) -> NpEnvState:
        # Advance timers and expose them in info BEFORE the base computes reward
        # (the air-time reward reads ctx.info["current_air_time"/"contact_time"]).
        self._update_contact_timers()
        state.info["current_air_time"] = self._current_air_time
        state.info["current_contact_time"] = self._current_contact_time
        state.info["last_air_time"] = self._last_air_time
        state.info["first_contact"] = self._first_contact
        return super().update_state(state)

    def reset(self, env_indices: np.ndarray):
        obs, info = super().reset(env_indices)
        ids = np.asarray(env_indices, dtype=np.intp)
        if self._current_air_time.shape[0] == self._num_envs:
            self._current_air_time[ids] = 0.0
            self._current_contact_time[ids] = 0.0
            self._last_air_time[ids] = 0.0
            self._first_contact[ids] = False
        return obs, info


registry.register_env("MicroduckVelocityFlat", MicroduckVelocityEnv, sim_backend="mujoco")
