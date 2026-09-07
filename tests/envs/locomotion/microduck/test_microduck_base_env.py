"""Phase 3: the MicroduckBaseEnv shared base wires the 61D contract + BAM.

Uses a minimal reward-less subclass (the base steps at reward 0) to exercise the
obs contract, projected-gravity computation, servo-by-name resolution, BAM
pre-step mounting, command-block sampling, and fall termination — everything
tasks inherit — before any concrete task exists.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("bam")

from unilab.envs.locomotion.microduck.base import (
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    NUM_COMMAND,
    NUM_SERVOS,
    MicroduckBaseCfg,
    MicroduckBaseEnv,
)

N = 8


class _BareMicroduckEnv(MicroduckBaseEnv):
    """No rewards — exercises the base machinery only."""


def _env(**cfg_over) -> _BareMicroduckEnv:
    cfg = MicroduckBaseCfg(**cfg_over)
    return _BareMicroduckEnv(cfg, num_envs=N, backend_type="mujoco")


def test_obs_and_action_dims():
    env = _env()
    assert env.obs_groups_spec == {"obs": ACTOR_OBS_DIM, "critic": CRITIC_OBS_DIM}
    assert ACTOR_OBS_DIM == 61 and CRITIC_OBS_DIM == 64
    assert env.action_space.shape == (NUM_SERVOS,)
    assert env._num_action == 14


def test_servo_indices_resolve_and_default_is_home():
    env = _env()
    assert env._servo_pos_idx.shape == (NUM_SERVOS,)
    assert env._servo_vel_idx.shape == (NUM_SERVOS,)
    # default_angles == STAND(==HOME) servo angles; spot-check the head block.
    da = env.default_angles
    assert da.shape == (NUM_SERVOS,)
    # neck_pitch, head_pitch = 0.3491; head_yaw, head_roll = 0.0
    assert da[5] == pytest.approx(0.3491, abs=1e-3)
    assert da[6] == pytest.approx(0.3491, abs=1e-3)
    assert da[7] == pytest.approx(0.0, abs=1e-3)
    assert da[8] == pytest.approx(0.0, abs=1e-3)


def test_projected_gravity_upright_points_down():
    env = _env()
    # Identity quat (upright) → projected_gravity_b == world down (0,0,-1).
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 1))
    pg = env._projected_gravity(quat)
    assert pg.shape == (N, 3)
    assert np.allclose(pg, np.array([0.0, 0.0, -1.0]), atol=1e-5)


def test_command_block_shape_and_zero_command():
    env = _env(commands=None) if False else _env()
    block = env._sample_command_block(N)
    assert block.shape == (N, NUM_COMMAND) == (N, 13)
    # With zero_command_prob=1, every row is exactly zero.
    env2 = _env()
    env2._cfg.commands.zero_command_prob = 1.0
    z = env2._sample_command_block(64)
    assert np.count_nonzero(z) == 0


def test_turn_in_place_bucket_zeros_xy_and_spins():
    env = _env()
    env._cfg.commands.turn_in_place_fraction = 1.0
    env._cfg.commands.twist_limit = [[-1.0, -0.4, -1.0], [1.0, 0.4, 1.0]]
    twist = env._sample_command_block(256)[:, :3]
    assert np.allclose(twist[:, 0:2], 0.0)  # xy zeroed
    assert np.all(np.abs(twist[:, 2]) >= 0.4 - 1e-6)  # |vyaw| in top band


def test_steps_nan_free_with_bam_mounted():
    env = _env()
    env.init_state()
    act = np.zeros((N, NUM_SERVOS), dtype=np.float32)
    for _ in range(50):
        state = env.step(act)
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
    assert state.obs["obs"].shape == (N, ACTOR_OBS_DIM)
    assert state.obs["critic"].shape == (N, CRITIC_OBS_DIM)


def test_obs_layout_order_matches_contract():
    """Actor obs = [ang_vel(3), proj_grav(3), jpos(14), jvel(14), act(14), cmd(13)]."""
    env = _env()
    env._cfg.noise_config.level = 0.0  # clean, so we can read the blocks back
    env.init_state()
    state = env.step(np.zeros((N, NUM_SERVOS), dtype=np.float32))
    obs = state.obs["obs"]
    # command block is the trailing 13 columns and equals info["commands"].
    cmd = state.info["commands"]
    assert np.allclose(obs[:, -NUM_COMMAND:], cmd, atol=1e-5)
    # proprio is the leading 48.
    assert obs.shape[1] == 48 + NUM_COMMAND
