"""microduck-specific reward functions (shared across the task family).

These complement ``envs.locomotion.common.rewards``. They take an explicit
parameter (sigma / std / weights) so the env binds them into ``RewardContext``
closures in ``_init_reward_functions`` — the dispatch itself only ever calls
``fn(ctx)``.

Sign convention (UniLab, not microduck_rl): reward terms return ≥0 and take a
POSITIVE weight; penalty terms return ≥0 (a cost) and take a NEGATIVE weight.
Every ``Episode_Reward/<penalty>`` must therefore read ≤0 in the logs.
"""

from __future__ import annotations

import numpy as np

from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.dtype_config import get_global_dtype

# Head/neck servos in the 14-servo order (neck_pitch, head_pitch, head_yaw,
# head_roll). The head_pose command tracks these four joints' delta from HOME.
HEAD_SERVO_IDX = np.array([5, 6, 7, 8])
# Leg servos (both legs) — everything that isn't head/neck.
LEG_SERVO_IDX = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])


def tracking_lin_vel(ctx: RewardContext, sigma: float) -> np.ndarray:
    """Exp reward for tracking commanded xy linear velocity (twist[:2])."""
    cmd = ctx.info["commands"]
    err = np.sum(np.square(cmd[:, :2] - ctx.linvel[:, :2]), axis=1)
    return np.asarray(np.exp(-err / sigma), dtype=get_global_dtype())


def tracking_ang_vel(ctx: RewardContext, sigma: float) -> np.ndarray:
    """Exp reward for tracking commanded yaw rate (twist[2]) vs body gyro z."""
    cmd = ctx.info["commands"]
    err = np.square(cmd[:, 2] - ctx.gyro[:, 2])
    return np.asarray(np.exp(-err / sigma), dtype=get_global_dtype())


def head_pose_tracking(ctx: RewardContext, std: float) -> np.ndarray:
    """Exp reward (mean over the 4 head joints) for tracking the head_pose command.

    The command (obs cols 3:7) is a per-joint target *delta from HOME*; the
    error is measured on the same view the policy sees (joint_pos − HOME), so
    the policy isn't punished for a quantity it can't observe (AGENTS.md).
    """
    cmd = ctx.info["commands"][:, 3:7]
    diff = ctx.dof_pos[:, HEAD_SERVO_IDX] - ctx.default_angles[HEAD_SERVO_IDX]
    per_joint = np.exp(-np.square((diff - cmd) / std))
    return np.asarray(np.mean(per_joint, axis=1), dtype=get_global_dtype())


def upright(ctx: RewardContext, std: float) -> np.ndarray:
    """Exp reward for uprightness from projected-gravity xy (0 when vertical)."""
    g = ctx.gravity
    assert g is not None
    xy_sq = np.sum(np.square(g[:, :2]), axis=1)
    return np.asarray(np.exp(-xy_sq / (std * std)), dtype=get_global_dtype())


def leg_pose_l2(ctx: RewardContext, weights: np.ndarray) -> np.ndarray:
    """Weighted L2 penalty (≥0) on leg-joint deviation from HOME.

    Head/neck joints carry weight 0 here — they are driven by
    ``head_pose_tracking``, so double-penalizing them toward HOME would fight
    the command (AGENTS.md pose/head conflict lesson).
    """
    diff = ctx.dof_pos - ctx.default_angles
    return np.asarray(np.sum(weights * np.square(diff), axis=1), dtype=get_global_dtype())


# ── foot-gait rewards (Phase 4.5: fix the skate, force real stepping) ──────────
# These read foot sensors that the env fetches and passes in (microduck keeps the
# reward functions pure); the gait-phase clock is read from ``ctx.info``.


def _feet_phase_targets(gait_phase: np.ndarray, swing_height: float) -> np.ndarray:
    """Per-foot swing-height target from the gait phase (cubic-bezier profile).

    Ported verbatim from g1's ``compute_feet_phase_height_targets``: each foot's
    target rises to ``swing_height`` over the swing half of its phase and returns
    to 0 over stance. ``gait_phase`` is ``(N, 2)`` in ``[0, 2π)``; returns
    ``(N, 2)`` height offsets (left, right) above the ground baseline.
    """

    def cubic_bezier_height(phi: np.ndarray) -> np.ndarray:
        phi_normalized = np.fmod(phi + np.pi, 2 * np.pi) - np.pi
        x = (phi_normalized + np.pi) / (2 * np.pi)

        def bezier(y_start: np.ndarray, y_end: np.ndarray, t: np.ndarray) -> np.ndarray:
            return y_start + (y_end - y_start) * (t**3 + 3 * (t**2 * (1 - t)))

        stance = bezier(np.zeros_like(x), np.full_like(x, swing_height), 2 * x)
        swing = bezier(np.full_like(x, swing_height), np.zeros_like(x), 2 * x - 1)
        return np.where(x <= 0.5, stance, swing)

    left = cubic_bezier_height(gait_phase[:, 0])
    right = cubic_bezier_height(gait_phase[:, 1])
    return np.column_stack([left, right])


def feet_phase(
    ctx: RewardContext,
    foot_z: np.ndarray,
    swing_height: float,
    ground_z: float,
    sigma: float,
    min_cmd_norm: float,
) -> np.ndarray:
    """Exp reward (≥0) for tracking the gait-clock swing-foot height — forces LIFT.

    ``foot_z`` is ``(N, 2)`` world Z of [left, right] foot sites; the swing target
    is ``ground_z + bezier(gait_phase)``. Gated by the commanded twist magnitude
    (``norm(commands[:, :3]) > min_cmd_norm``) so idle/zero-command envs aren't
    forced to step in place — this replaces g1's forward-speed gate to respect
    microduck's backward + explicit-idle command training.
    """
    gait_phase = ctx.info.get(
        "gait_phase", np.zeros((ctx.num_envs, 2), dtype=get_global_dtype())
    )
    targets = _feet_phase_targets(gait_phase, swing_height)  # (N, 2) heights above ground
    err = np.sum(np.square((foot_z - ground_z) - targets), axis=1)
    reward = np.exp(-err / sigma)
    cmd = ctx.info["commands"]
    moving = np.linalg.norm(cmd[:, :3], axis=1) > min_cmd_norm
    return np.asarray(reward * moving, dtype=get_global_dtype())


def feet_phase_contrast(
    ctx: RewardContext,
    foot_z: np.ndarray,
    swing_height: float,
    sigma: float,
    min_cmd_norm: float,
) -> np.ndarray:
    """Exp reward (≥0) for the L/R foot-HEIGHT DIFFERENCE tracking the gait clock.

    Ported from g1's ``_reward_feet_phase_contrast``. Where plain ``feet_phase``
    has a compromise basin (both feet held at a constant hover scores well), this
    rewards ``foot_z[L]-foot_z[R]`` matching ``target[L]-target[R]`` — i.e. one
    foot UP while the other is DOWN. A static/no-alternation gait can't satisfy
    it, so it breaks the shuffle optimum. Command-norm gated like ``feet_phase``.
    """
    gait_phase = ctx.info.get(
        "gait_phase", np.zeros((ctx.num_envs, 2), dtype=get_global_dtype())
    )
    targets = _feet_phase_targets(gait_phase, swing_height)
    actual_delta = foot_z[:, 0] - foot_z[:, 1]
    target_delta = targets[:, 0] - targets[:, 1]
    reward = np.exp(-np.square(actual_delta - target_delta) / sigma)
    cmd = ctx.info["commands"]
    moving = np.linalg.norm(cmd[:, :3], axis=1) > min_cmd_norm
    return np.asarray(reward * moving, dtype=get_global_dtype())


def feet_double_stance(
    ctx: RewardContext, contact: np.ndarray, min_cmd_norm: float
) -> np.ndarray:
    """Anti-drag penalty (≥0, cost): both feet on the ground while commanded to move.

    Ported from g1's ``_reward_feet_double_stance``. This is the missing "dragging
    is expensive" gradient: `feet_phase`/`air_time` reward stepping softly, but a
    stable double-stance shuffle can sit in a local optimum; charging a cost for
    every double-stance step (when a nonzero twist is commanded) pushes the policy
    out of that basin. Idle/zero-command envs are exempt (standing is correct
    there). ``contact`` is ``(N, 2)`` bool. Takes a NEGATIVE weight.
    """
    double = np.logical_and(contact[:, 0], contact[:, 1])
    cmd = ctx.info["commands"]
    moving = np.linalg.norm(cmd[:, :3], axis=1) > min_cmd_norm
    return np.asarray(double * moving, dtype=get_global_dtype())


def feet_air_time_landing(
    ctx: RewardContext,
    last_air_time: np.ndarray,
    first_contact: np.ndarray,
    threshold: float,
    cap: float,
    min_cmd_norm: float,
) -> np.ndarray:
    """Landing-settled air-time reward (SIGNED) — the anti-fast-shuffle driver.

    Settles ONCE per step, at the moment a foot touches down: it pays
    ``clip(last_air_time − threshold, −cap, +cap)`` for that foot, summed over any
    feet landing this step. A short swing (``last_air_time < threshold``) lands a
    NEGATIVE reward (penalises fast shuffling); a long swing lands POSITIVE
    (rewards slow, committed steps). The gradient is smooth on BOTH sides of
    ``threshold``, so the policy is pulled toward longer swings from wherever it
    currently is — unlike the hard-floor windowed reward, which was flat-zero
    (no gradient) below its lower bound and could never bootstrap a ~0.10 s gait
    over the 0.15 s cliff (see docs/microduck_port_plan.md, Run B/C).

    This is the IsaacLab / mjlab / microduck_rl ``feet_air_time`` formulation.
    NOTE: the return is SIGNED (can be < 0) yet carries a POSITIVE weight — it is
    a driver that rewards long steps and *penalises* short ones, so it is exempt
    from the "positive terms ≥ 0" log/test convention. ``last_air_time`` and
    ``first_contact`` are ``(N, 2)`` arrays the env stamps each step (the air time
    captured at touchdown; a bool of which feet landed this step). Command-norm
    gated so idle/zero-command envs aren't taxed for standing.
    """
    signed = np.clip(last_air_time - threshold, -cap, cap)
    reward = np.sum(signed * first_contact.astype(get_global_dtype()), axis=1)
    moving = np.linalg.norm(ctx.info["commands"][:, :2], axis=1) > min_cmd_norm
    return np.asarray(reward * moving, dtype=get_global_dtype())


def feet_slide(
    ctx: RewardContext, foot_speed_xy: np.ndarray, contact: np.ndarray
) -> np.ndarray:
    """Slip penalty (≥0, cost): horizontal foot speed summed over feet in contact.

    ``foot_speed_xy`` is ``(N, 2)`` world horizontal speed per foot, ``contact``
    is ``(N, 2)`` bool. A foot planted on the ground should not translate — this
    charges exactly the skate. It does NOT tax swing (contact-gated), so it's safe
    to keep on during gait discovery. Takes a NEGATIVE weight.
    """
    return np.asarray(np.sum(foot_speed_xy * contact, axis=1), dtype=get_global_dtype())
