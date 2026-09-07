"""Phase 4: MicroduckVelocityFlat builds, registers, and rewards correctly.

Exercises the full owner path: registry lookup → reward_config dict (as the
Hydra ``reward:`` block resolves) → env construction → step → per-term reward
signs. Penalties must be <=0 and the main tracking terms must be able to fire.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("bam")

from unilab.base import registry
from unilab.envs.locomotion.microduck.base import LEG_POSE_DEFAULT_WEIGHTS
from unilab.envs.locomotion.microduck.velocity import (
    MicroduckVelocityEnv,
    MicroduckVelocityFlatCfg,
)

N = 8

REWARD_DICT = {
    "scales": {
        "track_linear_velocity": 2.0,
        "track_angular_velocity": 2.0,
        "head_pose_tracking": 1.0,
        "upright": 2.0,
        "alive": 0.5,
        "feet_phase": 1.0,
        "feet_phase_contrast": 1.0,
        "feet_air_time": 3.0,
        "leg_pose": -0.5,
        "action_rate": -0.1,
        "lin_vel_z": -0.5,
        "ang_vel_xy": -0.05,
        "feet_slide": -0.1,
        "feet_double_stance": -1.5,
    },
    "lin_vel_sigma": 0.1,
    "ang_vel_sigma": 0.5,
    "head_pose_std": 0.5,
    "upright_std": 0.2236,
    "gait_frequency": 1.5,
    "feet_swing_height": 0.02,
    "feet_ground_z": 0.0,
    "feet_phase_sigma": 0.0004,
    "feet_slide_contact_threshold": 0.5,
    "feet_min_cmd_norm": 0.05,
    "feet_air_time_landing_threshold": 0.20,
    "feet_air_time_landing_cap": 0.15,
}

PENALTIES = ("leg_pose", "action_rate", "lin_vel_z", "ang_vel_xy", "feet_slide", "feet_double_stance")

FOOT_SENSORS = (
    "left_foot_pos", "right_foot_pos",
    "left_foot_vel", "right_foot_vel",
    "left_foot_contact", "right_foot_contact",
)


def _make() -> MicroduckVelocityEnv:
    return registry.make(
        "MicroduckVelocityFlat",
        sim_backend="mujoco",
        env_cfg_override={"reward_config": dict(REWARD_DICT)},
        num_envs=N,
    )


def test_registered_and_reward_config_constructed():
    assert "MicroduckVelocityFlat" in registry.list_envs() if hasattr(registry, "list_envs") else True
    env = _make()
    assert isinstance(env, MicroduckVelocityEnv)
    # reward dict resolved into the dataclass.
    assert env._reward_cfg.scales["track_linear_velocity"] == 2.0
    assert env._reward_cfg.lin_vel_sigma == 0.1
    # Every scale name has a bound reward fn.
    for name in REWARD_DICT["scales"]:
        assert name in env._reward_fns


def test_obs_and_action_contract():
    env = _make()
    assert env.obs_groups_spec == {"obs": 61, "critic": 64}
    assert env.action_space.shape == (14,)


def test_leg_pose_weights_zero_on_head():
    env = _make()
    w = env._leg_pose_weights
    assert np.array_equal(w, np.asarray(LEG_POSE_DEFAULT_WEIGHTS, dtype=np.float32))
    assert np.all(w[5:9] == 0.0)  # head/neck excluded
    assert np.all(w[[0, 1, 2, 3, 4, 9, 10, 11, 12, 13]] == 1.0)


def test_missing_reward_config_raises():
    cfg = MicroduckVelocityFlatCfg()  # reward_config None
    with pytest.raises(ValueError, match="reward_config"):
        MicroduckVelocityEnv(cfg, num_envs=2)


def test_step_and_reward_term_signs():
    env = _make()
    env.init_state()
    act = np.zeros((N, 14), dtype=np.float32)
    saw_log = {}
    for _ in range(24):  # enough to hit the reward log cadence
        state = env.step(act)
        assert np.isfinite(state.reward).all()
        log = state.info.get("log", {})
        for k, v in log.items():
            if k.startswith("Episode_Reward/"):
                saw_log[k] = v
    # Penalty terms must log <= 0 (weighted); positive terms are >= 0.
    for name in PENALTIES:
        key = f"Episode_Reward/{name}"
        if key in saw_log:
            assert saw_log[key] <= 1e-6, f"{key}={saw_log[key]} must be <=0"
    # NOTE: feet_air_time is SIGNED (landing reward penalises short swings), so it
    # is intentionally NOT asserted >=0 here.
    for name in ("upright", "alive", "feet_phase", "feet_phase_contrast"):
        key = f"Episode_Reward/{name}"
        if key in saw_log:
            assert saw_log[key] >= -1e-6, f"{key}={saw_log[key]} must be >=0"


def test_foot_sensors_resolve_on_walk_model():
    """The 6 foot sensors compile on scene_flat_motor.xml and the env helpers read them."""
    env = _make()
    for name in FOOT_SENSORS:
        data = env._backend.get_sensor_data(name)
        assert data.shape[0] == N, f"{name} wrong batch dim {data.shape}"
    env.init_state()
    env.step(np.zeros((N, 14), dtype=np.float32))
    assert env._foot_pos_z().shape == (N, 2)
    assert env._foot_speed_xy().shape == (N, 2)
    assert env._foot_contact().shape == (N, 2)
    assert env._foot_contact().dtype == np.bool_


def test_gait_phase_seeded_antiphase_and_advances():
    """gait_phase is in info after reset, seeded ~pi apart, and advances by delta/step."""
    env = _make()
    env.init_state()
    act = np.zeros((N, 14), dtype=np.float32)
    gp1 = env.step(act).info["gait_phase"].copy()
    gp2 = env.step(act).info["gait_phase"].copy()

    assert gp1.shape == (N, 2)
    assert np.all(gp1 >= 0.0) and np.all(gp1 < 2 * np.pi + 1e-6)
    # anti-phase: left/right seeded pi apart (offset_phase mode).
    diff = (gp1[:, 1] - gp1[:, 0]) % (2 * np.pi)
    assert np.allclose(diff, np.pi, atol=1e-3), f"not anti-phase: {diff}"
    # both columns advance by the per-step gait delta (mod 2pi).
    adv = (gp2 - gp1) % (2 * np.pi)
    assert np.allclose(adv, env._gait_phase_delta, atol=1e-4), f"adv={adv}"


def test_foot_reward_signs_weighted():
    """Positive foot rewards log >=0; feet_slide (cost) logs <=0 (all weighted)."""
    env = _make()
    env.init_state()
    env._enable_reward_log = True
    act = np.zeros((N, 14), dtype=np.float32)
    seen = {}
    for _ in range(8):  # dispatch logs every 4 steps
        state = env.step(act)
        seen.update(state.info.get("log", {}))
    for name in ("feet_phase", "feet_phase_contrast"):
        assert seen.get(f"reward/{name}", 0.0) >= -1e-6, (name, seen.get(f"reward/{name}"))
    assert seen.get("reward/feet_slide", 0.0) <= 1e-6, seen.get("reward/feet_slide")


def test_contact_timers_advance_and_feed_info():
    """Air/contact timers advance each step, are exposed in info, and are finite."""
    env = _make()
    env.init_state()
    act = np.zeros((N, 14), dtype=np.float32)
    state = env.step(act)
    assert env._current_air_time.shape == (N, 2)
    assert env._current_contact_time.shape == (N, 2)
    # exposed to the reward via info
    assert "current_air_time" in state.info
    assert "current_contact_time" in state.info
    # standing on both feet: contact time accrues, air time stays ~0
    for _ in range(10):
        state = env.step(act)
    ct = state.info["current_contact_time"]
    assert np.all(np.isfinite(ct))
    assert np.all(ct >= 0.0)
    # at least one foot has been in sustained contact (contact time grew past a step)
    assert ct.max() > env._cfg.ctrl_dt


def test_feet_air_time_landing_signed():
    """Landing air-time: short swing lands NEGATIVE, long lands POSITIVE, cap bounds it."""
    from types import SimpleNamespace

    from unilab.envs.locomotion.microduck.rewards import feet_air_time_landing

    thr, cap, min_cmd = 0.20, 0.15, 0.05
    cmd = np.tile(np.array([0.4, 0.0, 0.0], dtype=np.float32), (4, 1))  # moving

    # Row 0: fast step lands (air 0.10 < thr)     → negative (0.10-0.20 = -0.10)
    # Row 1: good step lands (air 0.30 > thr)      → positive (0.30-0.20 = +0.10)
    # Row 2: very long step lands (air 0.50)       → capped at +cap
    # Row 3: no landing this step (first_contact 0)→ 0 regardless of air time
    last_air = np.array([[0.10, 0.0], [0.30, 0.0], [0.50, 0.0], [0.40, 0.0]], dtype=np.float32)
    first_contact = np.array(
        [[True, False], [True, False], [True, False], [False, False]]
    )
    ctx = SimpleNamespace(num_envs=4, info={"commands": cmd})
    r = feet_air_time_landing(ctx, last_air, first_contact, thr, cap, min_cmd)
    assert abs(r[0] - (-0.10)) < 1e-6, f"fast step must land negative, got {r[0]}"
    assert abs(r[1] - (0.10)) < 1e-6, f"long step must land positive, got {r[1]}"
    assert abs(r[2] - cap) < 1e-6, f"very long step must cap at +{cap}, got {r[2]}"
    assert r[3] == 0.0, f"no landing → 0, got {r[3]}"

    # Idle (zero command) is exempt even on a good landing.
    ctx.info["commands"] = np.zeros((4, 3), dtype=np.float32)
    assert np.all(feet_air_time_landing(ctx, last_air, first_contact, thr, cap, min_cmd) == 0.0)


def test_feet_air_time_landing_stamped_in_info():
    """The env stamps last_air_time / first_contact each step for the landing reward."""
    env = _make()
    env.init_state()
    state = env.step(np.zeros((N, 14), dtype=np.float32))
    assert env._last_air_time.shape == (N, 2)
    assert env._first_contact.shape == (N, 2)
    assert env._first_contact.dtype == np.bool_
    assert "last_air_time" in state.info
    assert "first_contact" in state.info


def test_foot_rewards_registered():
    env = _make()
    for name in ("feet_phase", "feet_phase_contrast", "feet_air_time",
                 "feet_slide", "feet_double_stance"):
        assert name in env._reward_fns


def test_curriculum_switches_to_stage_b():
    """At the Stage-A→B boundary the env swaps in the refine weights (once)."""
    rd = dict(REWARD_DICT)
    rd["scales"] = dict(REWARD_DICT["scales"])  # don't mutate the shared dict
    rd["curriculum_enabled"] = True
    rd["curriculum_stage_b_iter"] = 1
    rd["curriculum_num_steps_per_env"] = 1  # boundary = step 1
    env = registry.make(
        "MicroduckVelocityFlat", sim_backend="mujoco",
        env_cfg_override={"reward_config": rd}, num_envs=N,
    )
    env.init_state()
    # Stage A initially.
    assert not env._stage_b_applied
    assert env._reward_cfg.scales["feet_double_stance"] == -1.5
    assert env._reward_cfg.scales["feet_phase"] == 1.0
    delta_a = env._gait_phase_delta

    act = np.zeros((N, 14), dtype=np.float32)
    for _ in range(3):
        env.step(act)

    # Stage B applied: double-stance penalty RELAXED (-1.5 → -0.5, keeps the
    # "leave double support" pressure), higher lift, slower clock.
    assert env._stage_b_applied
    assert env._reward_cfg.scales["feet_double_stance"] == -0.5
    assert env._reward_cfg.scales["feet_phase"] == 3.0
    assert env._reward_cfg.scales["feet_air_time"] == 10.0
    assert env._reward_cfg.feet_swing_height == 0.03
    assert env._gait_phase_delta < delta_a  # 1.0 Hz < 1.5 Hz → smaller step
