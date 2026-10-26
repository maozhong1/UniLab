"""H2 SONIC feet_slide: the gate follows the reference stance only, whatever the robot foot does."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.envs.motion_tracking.h2.tracking_sonic import (
    H2SonicMotionTrackingEnv,
    SonicRewardConfig,
)


def _env(num_envs: int) -> H2SonicMotionTrackingEnv:
    env = object.__new__(H2SonicMotionTrackingEnv)
    env._strict_foot_indices = np.array([0, 1], dtype=np.intp)
    env._cfg = SimpleNamespace(reward_config=SonicRewardConfig())
    nf = 2
    env._foot_xy_sq = np.empty((num_envs, nf, 2))
    env._foot_perfoot = np.empty((num_envs, nf))
    env._foot_ref_speed_sq = np.empty((num_envs, nf))
    env._foot_gate = np.empty((num_envs, nf), dtype=bool)
    env._foot_gate_tmp = np.empty((num_envs, nf), dtype=bool)
    env._foot_penalty_slide = np.empty((num_envs,))
    return env


def test_feet_slide_penalizes_a_lifted_robot_foot_during_reference_stance() -> None:
    rc = SonicRewardConfig()
    h, v = rc.feet_contact_height, rc.feet_ref_stance_speed
    # env 0: reference stands, robot left foot lifted well above the contact height and moving
    # env 1: reference left foot swinging (moving) -> left foot not gated
    # env 2: reference left foot high (swing apex) -> not gated
    ref_pos = np.zeros((3, 2, 3))
    ref_pos[:, :, 2] = 0.05
    ref_pos[2, 0, 2] = h + 0.05
    ref_vel = np.zeros((3, 2, 3))
    ref_vel[1, 0, 0] = 2.0 * v
    robot_pos = np.zeros((3, 2, 3))
    robot_pos[:, :, 2] = 0.05
    robot_pos[0, 0, 2] = h + 0.10
    robot_vel = np.zeros((3, 2, 3))
    robot_vel[:, 0, :2] = (0.3, 0.4)  # left foot 0.5 m/s in every env
    robot_vel[:, 1, :2] = (0.0, 0.2)  # right foot 0.2 m/s, reference stance everywhere

    env = _env(3)
    ctx = SimpleNamespace(
        motion_data=SimpleNamespace(body_pos_w=ref_pos, body_lin_vel_w=ref_vel),
        robot_body_pos_w=robot_pos,
        robot_body_lin_vel_w=robot_vel,
    )
    out = env._reward_feet_slide(ctx)
    np.testing.assert_allclose(out, [(0.25 + 0.04) / 2, 0.04 / 2, 0.04 / 2], atol=1e-12)


def test_feet_slide_is_off_by_default() -> None:
    assert SonicRewardConfig().scales["feet_slide"] == 0.0
