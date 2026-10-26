"""H2 SONIC joint_soft_limit / joint_target_limit rewards and leg PD gain consistency."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs.motion_tracking.h2.tracking_sonic import (
    _ANKLE_PITCH_KP,
    _HIP_KNEE_KP,
    H2SonicMotionTrackingEnv,
    SonicRewardConfig,
)


def _env_with_bounds(lower: np.ndarray, upper: np.ndarray, factor: float):
    env = object.__new__(H2SonicMotionTrackingEnv)
    env._joint_lower, env._joint_upper = lower, upper
    mid, half = 0.5 * (lower + upper), 0.5 * factor * (upper - lower)
    env._soft_joint_lower, env._soft_joint_upper = mid - half, mid + half
    return env


def test_joint_soft_limit_sums_excess_beyond_centred_soft_range() -> None:
    lower = np.array([-0.3, -0.1, -1.0])
    upper = np.array([0.35, 2.5, 1.0])
    env = _env_with_bounds(lower, upper, factor=0.9)
    soft_lo, soft_hi = env._soft_joint_lower, env._soft_joint_upper
    dof_pos = np.array([
        [0.0, 1.0, 0.0],                          # inside: no penalty
        [upper[0] + 0.04, lower[1] - 0.01, 0.0],  # past both hard stops
        [soft_hi[0] - 0.01, soft_lo[1] + 0.01, soft_hi[2] + 0.02],
    ])
    ctx = SimpleNamespace(
        dof_pos=dof_pos,
        joint_error=np.empty_like(dof_pos),
        joint_error_upper=np.empty_like(dof_pos),
        reward_term=np.empty(3),
    )
    out = env._reward_joint_soft_limit(ctx)
    expected = [0.0, (upper[0] + 0.04 - soft_hi[0]) + (soft_lo[1] - lower[1] + 0.01), 0.02]
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_joint_target_limit_sums_pd_target_excess_beyond_hard_range() -> None:
    lower = np.array([-0.3, -0.1])
    upper = np.array([0.35, 2.5])
    env = _env_with_bounds(lower, upper, factor=0.95)
    scale, default = np.array([0.1, 0.5]), np.array([0.0, 0.6])
    env._cfg = SimpleNamespace(control_config=SimpleNamespace(action_scale=scale))
    env._effective_default_angles = lambda: default
    actions = np.array([
        [1.0, 0.0],    # targets 0.1, 0.6: inside
        [16.0, -2.0],  # targets 1.6 (> 0.35), -0.4 (< -0.1)
    ])
    dof_pos = np.zeros_like(actions)
    ctx = SimpleNamespace(
        info={"current_actions": actions},
        dof_pos=dof_pos,
        joint_error=np.empty_like(dof_pos),
        joint_error_upper=np.empty_like(dof_pos),
        reward_term=np.empty(2),
    )
    out = env._reward_joint_target_limit(ctx)
    np.testing.assert_allclose(out, [0.0, (1.6 - 0.35) + (-0.1 + 0.4)], atol=1e-12)


def test_joint_limit_terms_are_off_by_default() -> None:
    cfg = SonicRewardConfig()
    assert cfg.scales["joint_soft_limit"] == 0.0
    assert cfg.scales["joint_target_limit"] == 0.0
    assert 0.0 < cfg.soft_joint_limit_factor <= 1.0


def test_leg_gain_constants_match_h2_xml() -> None:
    # _HIP_KNEE_KP / _ANKLE_PITCH_KP define the leg action scale and must match the
    # <position kp> actuators the policy is trained (and deployed) with.
    root = ET.parse(ASSETS_ROOT_PATH / "robots" / "h2" / "h2.xml").getroot()
    kp = {a.get("joint"): float(a.get("kp")) for a in root.iter("position")}
    for side in ("left", "right"):
        legs = [kp[f"{side}_{j}_joint"] for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee")]
        np.testing.assert_allclose(legs, _HIP_KNEE_KP)
        assert kp[f"{side}_ankle_pitch_joint"] == _ANKLE_PITCH_KP
