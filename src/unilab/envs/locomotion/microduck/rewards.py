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


def _feet_gait_targets(
    gait_phase: np.ndarray, swing_height: float, duty: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-foot ``(swing-height target, stance-expected mask)`` from a duty-cycle clock.

    The original g1 bezier bumped up-and-down EVERY cycle with no planted-stance
    region, so a fast 5 Hz tap could partially satisfy a 1 Hz clock and never
    commit to a walk (see docs/microduck_port_plan.md, Run D "fast mincing"). This
    splits each 2π cycle into a STANCE fraction ``duty`` (foot planted, target
    height 0) and a SWING fraction ``1-duty`` (a single cubic-bezier lift
    0→swing_height→0). ``gait_phase`` is ``(N, 2)`` in ``[0, 2π)``; returns
    ``height (N, 2)`` above the ground baseline and ``stance (N, 2)`` bool
    (True where the foot is scheduled to be planted).
    """
    x = np.asarray(gait_phase, dtype=get_global_dtype()) / (2.0 * np.pi)  # [0, 1)
    stance = x < duty
    s = np.clip((x - duty) / max(1.0 - duty, 1e-6), 0.0, 1.0)  # swing progress [0,1]

    def bezier(y0: float, y1, t: np.ndarray) -> np.ndarray:
        return y0 + (y1 - y0) * (t**3 + 3.0 * (t**2 * (1.0 - t)))

    up = bezier(0.0, swing_height, 2.0 * s)
    down = bezier(swing_height, 0.0, 2.0 * s - 1.0)
    bump = np.where(s <= 0.5, up, down)
    height = np.where(stance, 0.0, bump)
    return height.astype(get_global_dtype()), stance


def feet_phase(
    ctx: RewardContext,
    foot_z: np.ndarray,
    swing_height: float,
    ground_z: float,
    sigma: float,
    min_cmd_norm: float,
    duty: float,
) -> np.ndarray:
    """Exp reward (≥0) for tracking the duty-cycle swing-foot height — LIFT in swing,
    PLANTED (target 0) in stance.

    ``foot_z`` is ``(N, 2)`` world Z of [left, right] foot sites; the target is
    ``ground_z + _feet_gait_targets(...)`` (0 during the scheduled stance, a
    bezier lift during swing). Gated by the commanded twist magnitude so
    idle/zero-command envs aren't forced to step in place.
    """
    gait_phase = ctx.info.get(
        "gait_phase", np.zeros((ctx.num_envs, 2), dtype=get_global_dtype())
    )
    targets, _ = _feet_gait_targets(gait_phase, swing_height, duty)  # (N, 2) heights
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
    duty: float,
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
    targets, _ = _feet_gait_targets(gait_phase, swing_height, duty)
    actual_delta = foot_z[:, 0] - foot_z[:, 1]
    target_delta = targets[:, 0] - targets[:, 1]
    reward = np.exp(-np.square(actual_delta - target_delta) / sigma)
    cmd = ctx.info["commands"]
    moving = np.linalg.norm(cmd[:, :3], axis=1) > min_cmd_norm
    return np.asarray(reward * moving, dtype=get_global_dtype())


def feet_contact_schedule(
    ctx: RewardContext, contact: np.ndarray, duty: float, min_cmd_norm: float
) -> np.ndarray:
    """Dense reward (≥0) — foot CONTACT state matches the duty-cycle clock.

    THE cadence lever. Pays the fraction of feet whose ground-contact agrees with
    the clock schedule (planted during the ``duty`` stance fraction, airborne
    during swing). A fast shuffle whose contacts are out of phase with the (slow,
    1 Hz) clock scores low, so syncing to the clock — i.e. slowing the cadence and
    holding each stance/swing for its full duration — is the only way to max it.
    This is the piece missing in Runs A–D: nothing tied CONTACT (hence cadence) to
    the clock, only instantaneous height. Standing still scores only the stance
    fraction (mismatched during every swing window), so it's not farmable by not
    stepping. Command-gated (idle exempt).
    """
    gait_phase = ctx.info.get(
        "gait_phase", np.zeros((ctx.num_envs, 2), dtype=get_global_dtype())
    )
    _, stance = _feet_gait_targets(gait_phase, 0.0, duty)  # only the mask matters here
    match = np.asarray(contact, dtype=bool) == stance  # planted in stance, air in swing
    reward = np.mean(match.astype(get_global_dtype()), axis=1)
    moving = np.linalg.norm(ctx.info["commands"][:, :3], axis=1) > min_cmd_norm
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


def feet_clearance(
    ctx: RewardContext,
    foot_z: np.ndarray,
    foot_speed_xy: np.ndarray,
    ground_z: float,
    target: float,
) -> np.ndarray:
    """Clock-FREE swing-clearance cost (≥0): squared foot-height error × horizontal
    foot speed, summed over feet.

    ``err = ((foot_z − ground_z) − target)² · horizontal_speed``. A PLANTED foot
    (speed ≈ 0) is never charged; a foot moving horizontally (mid-swing) is pulled
    toward the apex ``target``. This sets step HEIGHT without any gait clock and
    without rewarding a parked-up foot, and it is NEUTRAL on cadence, so it does
    not fight ``feet_air_time`` (the clock-free cadence driver). Paired: air-time
    picks the swing LENGTH (slow), clearance picks the swing HEIGHT. Ported from
    IsaacLab's ``feet_clearance``. Takes a NEGATIVE weight.
    """
    err = np.square((foot_z - ground_z) - target)  # (N, 2)
    return np.asarray(np.sum(err * foot_speed_xy, axis=1), dtype=get_global_dtype())


# ── roller stride / swizzle rewards (ported 1:1 from microduck_rl mdp.py) ───────
# The stride gait is CLOCK-FREE: it reads REAL per-foot air/contact times (env
# timers), never a gait-phase clock (that path failed in velocity — see port
# plan). Env fetches sensor/wheel/torque arrays and passes them in; funcs stay
# pure. Sign convention: rewards ≥0 (positive weight), penalties ≥0 (negative
# weight); a few (single_support, com_height_target, leg_symmetry) are signed.


def _dt() -> type:
    return get_global_dtype()


def _cmd(ctx: RewardContext) -> np.ndarray:
    return ctx.info["commands"]


def _forward_gate(ctx: RewardContext, v_ref: float) -> np.ndarray | None:
    """0→1 ramp in body forward speed; None when disabled (v_ref<=0)."""
    if v_ref <= 0.0:
        return None
    v = ctx.linvel[:, 0]
    return np.asarray(np.clip(np.clip(v, 0.0, None) / v_ref, None, 1.0), dtype=_dt())


def wheel_speed_reward(
    ctx: RewardContext,
    forward_omega: np.ndarray,
    vel_scale: float,
    wheel_radius: float = 0.0175,
    bidirectional: bool = False,
) -> np.ndarray:
    """Reward wheel spin in the commanded direction (tanh-saturated). Sole task
    reward; ``forward_omega`` is the mean of the 4 wheel dof velocities."""
    cmd_x = _cmd(ctx)[:, 0]
    omega_scale = vel_scale / wheel_radius
    if bidirectional:
        aligned = np.sign(cmd_x) * forward_omega
        out = np.abs(cmd_x) * np.tanh(np.clip(aligned, 0.0, None) / omega_scale)
    else:
        out = np.clip(cmd_x, 0.0, None) * np.tanh(np.clip(forward_omega, 0.0, None) / omega_scale)
    return np.asarray(out, dtype=_dt())


def braking_reward(ctx: RewardContext, vel_std: float = 0.3) -> np.ndarray:
    """Reward stopping when cmd_x<0; silent when cmd_x>=0."""
    cmd_x = _cmd(ctx)[:, 0]
    fwd = ctx.linvel[:, 0]
    stopped = np.exp(-np.square(np.clip(fwd, 0.0, None)) / (vel_std**2))
    return np.asarray(np.clip(-cmd_x, 0.0, None) * stopped, dtype=_dt())


def skating_air_time_reward(
    ctx: RewardContext,
    air_time: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    vel_gate_ref: float,
) -> np.ndarray:
    """Reward each foot's air time inside [min,max] while pushing forward."""
    in_range = (air_time > threshold_min) & (air_time < threshold_max)
    reward = np.sum(in_range.astype(_dt()), axis=1) * np.clip(_cmd(ctx)[:, 0], 0.0, None)
    gate = _forward_gate(ctx, vel_gate_ref)
    if gate is not None:
        reward = reward * gate
    return np.asarray(reward, dtype=_dt())


def single_support_reward(
    ctx: RewardContext,
    contact: np.ndarray,
    vel_gate_ref: float,
    double_penalty: float = 0.25,
) -> np.ndarray:
    """+cmd_x·gate for exactly one blade down; −double_penalty·cmd_x for both
    down (SIGNED → positive weight). ``contact`` is (N,2) bool."""
    n = np.sum(contact.astype(_dt()), axis=1)
    cmd_x = np.clip(_cmd(ctx)[:, 0], 0.0, None)
    single_r = (n == 1).astype(_dt()) * cmd_x
    gate = _forward_gate(ctx, vel_gate_ref)
    if gate is not None:
        single_r = single_r * gate
    return np.asarray(single_r - double_penalty * (n >= 2).astype(_dt()) * cmd_x, dtype=_dt())


def glide_reward(
    ctx: RewardContext,
    contact: np.ndarray,
    leg_joint_vel_sq: np.ndarray,
    vel_ref: float = 0.2,
    stillness_std: float = 5.0,
) -> np.ndarray:
    """Reward coasting on ONE blade with quiet legs while moving forward."""
    single = (np.sum(contact.astype(_dt()), axis=1) == 1).astype(_dt())
    gate = _forward_gate(ctx, vel_ref)
    if gate is None:
        gate = np.ones((ctx.num_envs,), dtype=_dt())
    stillness = np.exp(-leg_joint_vel_sq / (stillness_std**2))
    active = (_cmd(ctx)[:, 0] >= 0.0).astype(_dt())
    return np.asarray(single * gate * stillness * active, dtype=_dt())


def gait_symmetry_penalty(ctx: RewardContext, swing_accum: np.ndarray) -> np.ndarray:
    """Penalise cumulative L/R swing-time imbalance |L−R|/(L+R). (N,2) accum."""
    L, R = swing_accum[:, 0], swing_accum[:, 1]
    return np.asarray(np.abs(L - R) / (L + R + 1e-3), dtype=_dt())


def grounded_reward(ctx: RewardContext, contact: np.ndarray) -> np.ndarray:
    """Swizzle: reward BOTH blades down, scaled by |cmd_x| (bidirectional)."""
    grounded = (np.sum(contact.astype(_dt()), axis=1) >= 2).astype(_dt())
    return np.asarray(grounded * np.abs(_cmd(ctx)[:, 0]), dtype=_dt())


def leg_symmetry_reward(ctx: RewardContext, left_q: np.ndarray, right_q: np.ndarray) -> np.ndarray:
    """Swizzle: −mean|q_left+q_right| over matched leg pairs (SIGNED, pos weight)."""
    return np.asarray(-np.mean(np.abs(left_q + right_q), axis=1), dtype=_dt())


def com_height_target(ctx: RewardContext, target_min: float, target_max: float) -> np.ndarray:
    """+1 in [min,max], −squared distance outside (SIGNED, positive weight)."""
    h = np.nan_to_num(ctx.base_height, nan=0.0)
    below, above = h < target_min, h > target_max
    pen = np.square(h - target_min) * below + np.square(h - target_max) * above
    return np.asarray((~(below | above)).astype(_dt()) - pen, dtype=_dt())


def forward_lean_reward(ctx: RewardContext, target_pitch: float, std: float) -> np.ndarray:
    """Reward slight forward lean (projected_gravity_x) while pushing."""
    lean = ctx.gravity[:, 0]  # type: ignore[index]
    push = np.clip(_cmd(ctx)[:, 0], 0.0, None)
    return np.asarray(push * np.exp(-np.square(lean - target_pitch) / (std**2)), dtype=_dt())


def heading_hold_reward(ctx: RewardContext, heading_err: np.ndarray, std: float) -> np.ndarray:
    """Reward holding spawn heading; env passes the wrapped yaw error."""
    return np.asarray(np.exp(-np.square(heading_err) / (std**2)), dtype=_dt())


def feet_flat_penalty(ctx: RewardContext, foot_tilt_xy: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """Penalise stance-blade tilt: per-foot gravity-in-site xy², contact-gated."""
    return np.asarray(np.sum(foot_tilt_xy * contact.astype(_dt()), axis=1), dtype=_dt())


def neck_action_rate_l2(ctx: RewardContext) -> np.ndarray:
    """Penalise neck/head (idx 5-8) action rate."""
    cur = ctx.info["current_actions"][:, 5:9]
    last = ctx.info["last_actions"][:, 5:9]
    return np.asarray(np.sum(np.square(cur - last), axis=1), dtype=_dt())


def variable_posture(
    ctx: RewardContext,
    std_standing: np.ndarray,
    std_walking: np.ndarray,
    std_running: np.ndarray,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
) -> np.ndarray:
    """Speed-regime pose reward exp(−mean(err²/std²)) over the 14 servos. std_*
    are per-servo (14,) vectors resolved from the cfg regex dicts by the env."""
    cmd = _cmd(ctx)
    speed = np.linalg.norm(cmd[:, :2], axis=1) + np.abs(cmd[:, 2])
    standing = (speed < walking_threshold)[:, None]
    walking = ((speed >= walking_threshold) & (speed < running_threshold))[:, None]
    running = (speed >= running_threshold)[:, None]
    std = std_standing * standing + std_walking * walking + std_running * running
    err_sq = np.square(ctx.dof_pos - ctx.default_angles)
    return np.asarray(np.exp(-np.mean(err_sq / (std**2), axis=1)), dtype=_dt())


def action_over_limit_penalty(
    ctx: RewardContext, target: np.ndarray, joint_range: np.ndarray, overshoot: float = 0.3
) -> np.ndarray:
    """Penalise commanded targets beyond (hard limit + overshoot). target (N,14),
    joint_range (14,2) hard limits."""
    lo = joint_range[:, 0] - overshoot
    hi = joint_range[:, 1] + overshoot
    over = np.clip(target - hi, 0.0, None) + np.clip(lo - target, 0.0, None)
    return np.asarray(np.sum(over, axis=1), dtype=_dt())


def angular_momentum_penalty(ctx: RewardContext, angmom: np.ndarray) -> np.ndarray:
    """Penalise whole-body angular momentum (root_angmom sensor, N×3)."""
    return np.asarray(np.sum(np.square(angmom), axis=1), dtype=_dt())
