"""S6 cfg-invariant tests for the roller STRIDE task (MicroduckRollers).

CPU/MuJoCo, no GPU. Locks: build dims (61D obs / 14 action), wheel + servo index
resolution, the 61D command layout (heading-off yaw + zero head/body slots),
reward-term signs (every penalty <=0; signed terms exempt), and the action_rate
curriculum step function.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

from unilab.envs.locomotion.microduck.rollers import (  # noqa: E402
    MicroduckRollersCfg,
    MicroduckRollersEnv,
    MicroduckRollersRewardCfg,
    WHEEL_NAMES,
)

# Negative-weight penalties whose weighted log MUST be <=0. single_support,
# com_height_target and (swizzle-only) leg_symmetry are SIGNED → exempt.
PENALTY_TERMS = {
    "body_ang_vel", "angular_momentum", "action_rate", "self_collisions",
    "feet_flat", "neck_action_rate", "neck_joint_pos", "joint_torques", "action_over_limit",
    "hip_roll_neutral", "gait_symmetry",
}
POSITIVE_TERMS = {"wheel_speed", "braking", "skating_air_time", "glide",
                  "forward_lean", "heading_hold", "upright", "pose"}

SCALES = {
    "wheel_speed": 10.0, "braking": 1.0, "skating_air_time": 1.5, "glide": 4.0,
    "single_support": 3.0, "forward_lean": 1.5, "heading_hold": 1.0,
    "com_height_target": 2.0, "upright": 2.0, "pose": 2.0,
    "body_ang_vel": -0.05, "angular_momentum": -0.02, "action_rate": -1.0,
    "self_collisions": -1.0, "feet_flat": -2.0, "neck_action_rate": -0.5, "neck_joint_pos": -0.5,
    "joint_torques": -0.001, "action_over_limit": -0.5, "hip_roll_neutral": -2.0,
    "gait_symmetry": -1.0,
}


def _env(num_envs: int = 4) -> MicroduckRollersEnv:
    cfg = MicroduckRollersCfg(reward_config=MicroduckRollersRewardCfg(scales=dict(SCALES)))
    return MicroduckRollersEnv(cfg, num_envs=num_envs)


def test_build_dims_and_indices() -> None:
    env = _env()
    obs, _ = env.reset(np.arange(4))
    assert obs["obs"].shape == (4, 61)
    assert obs["critic"].shape == (4, 64)
    assert env.action_space.shape == (14,)
    # 4 passive wheels resolve by name; servo hard limits are (14, 2).
    assert env._wheel_vel_idx.shape == (4,)
    assert len(WHEEL_NAMES) == 4
    assert env._servo_joint_range.shape == (14, 2)


def test_default_angles_follow_stand_by_servo_name() -> None:
    # Passive wheels interleave in qpos on the roller model, so HOME must be
    # resolved by servo name, not by slicing the keyframe tail.
    import mujoco

    from unilab.envs.locomotion.microduck.base import SERVO_NAMES

    env = _env(num_envs=1)
    model = mujoco.MjModel.from_xml_path(env.cfg.scene.model_file)
    key = model.key("STAND").id
    expected = np.array(
        [model.key_qpos[key, model.jnt_qposadr[model.joint(n).id]] for n in SERVO_NAMES]
    )
    np.testing.assert_allclose(env.default_angles, expected, atol=1e-6)


def test_reset_spawns_at_source_height_and_at_rest() -> None:
    # Source reset_base: z ~ U(0.1335, 0.1435) (STAND z=0.12 sinks the tires
    # ~2 cm into the floor) and zero base velocity.
    import mujoco

    env = _env(num_envs=64)
    env.reset(np.arange(64))
    model = mujoco.MjModel.from_xml_path(env.cfg.scene.model_file)
    snap = np.asarray(env.get_physics_state_snapshot())
    qpos, qvel = snap[:, 1 : 1 + model.nq], snap[:, 1 + model.nq : 1 + model.nq + model.nv]
    assert np.all(qpos[:, 2] >= 0.1335 - 1e-6) and np.all(qpos[:, 2] <= 0.1435 + 1e-6)
    np.testing.assert_allclose(qvel[:, 0:6], 0.0, atol=1e-6)


def test_forward_bucket_and_mid_episode_resampling() -> None:
    env = _env(num_envs=4)
    cmd_cfg = env.cfg.commands
    assert cmd_cfg.forward_fraction == 0.2 and cmd_cfg.resampling_time_range == [3.0, 8.0]

    np.random.seed(0)
    block = env._sample_command_block(20000)
    frac_fwd = np.mean(block[:, 0] >= 0.3)
    assert 0.38 < frac_fwd < 0.46  # 0.8 * P(U(-.5,.6) >= .3) + 0.2 ~= 0.42

    env.init_state()
    assert np.all((env._cmd_time_left >= 3.0) & (env._cmd_time_left <= 8.0))
    env._cmd_time_left[:] = [0.01, 5.0, 5.0, 5.0]
    before = env.state.info["commands"].copy()
    env.step(np.zeros((4, 14), dtype=np.float32))
    after = env.state.info["commands"]
    assert not np.array_equal(after[0], before[0]), "expired env must resample"
    np.testing.assert_array_equal(after[1:], before[1:])
    assert 3.0 <= env._cmd_time_left[0] <= 8.0


def test_command_layout_heading_off_and_zero_head_body() -> None:
    """Stride obs: twist yaw (col 50) == 0 (heading held via reward, not command),
    head (51:55) and body (55:61) slots zero-padded (roller deploy state)."""
    env = _env()
    obs, _ = env.reset(np.arange(4))
    cmd = obs["obs"][:, 48:61]  # 13D command block
    assert np.allclose(cmd[:, 2], 0.0), "twist yaw must be 0 (heading off)"
    assert np.allclose(cmd[:, 3:], 0.0), "head+body command slots must be 0"
    # vx is sampled in (-0.5, 0.6)
    assert np.all(cmd[:, 0] >= -0.5) and np.all(cmd[:, 0] <= 0.6)


def test_reward_signs() -> None:
    env = _env()
    env.reset(np.arange(4))
    for _ in range(8):  # reach a logging step (every 4)
        st = env.step(np.zeros((4, 14), dtype=np.float32))
    log = st.info["log"]
    for term in PENALTY_TERMS:
        key = f"reward/{term}"
        if key in log:
            assert log[key] <= 1e-6, f"penalty {term} weighted > 0: {log[key]}"
    for term in POSITIVE_TERMS:
        key = f"reward/{term}"
        if key in log:
            assert log[key] >= -1e-6, f"positive {term} weighted < 0: {log[key]}"
    assert np.isfinite(st.reward).all()


def test_action_rate_curriculum_steps() -> None:
    env = _env()
    rc = env._reward_cfg
    npe = rc.curriculum_num_steps_per_env
    env.step_counter = 0
    env._maybe_apply_curriculum()
    assert rc.scales["action_rate"] == -1.0
    env.step_counter = rc.action_rate_stage2_iter * npe
    env._maybe_apply_curriculum()
    assert rc.scales["action_rate"] == -1.5
    env.step_counter = rc.action_rate_stage3_iter * npe
    env._maybe_apply_curriculum()
    assert rc.scales["action_rate"] == -2.0
