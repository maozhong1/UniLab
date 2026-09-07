"""Phase 1 smoke: the ported microduck MJCF assets load in the MuJoCo backend.

Verifies the three robot models copied from microduck_rl
(walk / groundcontact / rollers) materialize in `mujoco_uni`, expose the
14-servo actuator layout in the canonical order, carry the proprioceptive
sensors the obs contract needs, and step NaN-free from the STAND keyframe.
No env/reward/actuator-model logic yet — that is Phase 2+.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.scene import SceneCfg

pytest.importorskip("mujoco", reason="mujoco not installed")

BASE_NAME = "trunk_base"
NUM_ENVS = 2
SIM_DT = 0.005

# ctrl idx = joint idx on walk/groundcontact models (AGENTS.md invariant):
# 0-4 left leg, 5-8 neck/head, 9-13 right leg.
EXPECTED_ACTUATORS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
# Sensors the 61D proprioception block reads (from microduck sensors.xml).
EXPECTED_SENSORS = {"orientation", "imu_ang_vel", "imu_lin_vel", "imu_accel", "root_angmom"}


def _scene(name: str) -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "microduck" / name)


# (scene file, expected nq, expected nv, expected passive wheel joints)
MODELS = [
    pytest.param("scene_flat.xml", 21, 20, [], id="walk"),
    pytest.param("scene_groundcontact.xml", 21, 20, [], id="groundcontact"),
    pytest.param(
        "scene_rollers.xml", 25, 24,
        ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"],
        id="rollers",
    ),
]


@pytest.mark.parametrize("scene,nq,nv,passive", MODELS)
def test_microduck_model_loads(scene: str, nq: int, nv: int, passive: list[str]) -> None:
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    bkd = MuJoCoBackend(SceneCfg(model_file=_scene(scene)), NUM_ENVS, SIM_DT, base_name=BASE_NAME)
    bkd.materialize()
    m = cast(Any, bkd.model)

    assert (m.nq, m.nv, m.nu) == (nq, nv, 14), f"{scene}: dims {(m.nq, m.nv, m.nu)}"
    assert bkd.num_actuators == 14

    names = [m.actuator(i).name for i in range(m.nu)]
    assert names == EXPECTED_ACTUATORS, f"{scene}: actuator order {names}"

    sensors = {m.sensor(i).name for i in range(m.nsensor)}
    assert EXPECTED_SENSORS.issubset(sensors), f"{scene}: missing {EXPECTED_SENSORS - sensors}"

    joints = [m.joint(i).name for i in range(m.njnt)]
    assert [j for j in joints if j.startswith("passive_")] == passive

    # Unactuated joints must all be passive_* (selector invariant).
    hinge_servos = [n for n in names]
    assert all(not s.startswith("passive_") for s in hinge_servos)

    bkd.step(np.zeros((NUM_ENVS, m.nu)), nsteps=4)
    assert np.isfinite(bkd.get_base_pos()).all(), f"{scene}: non-finite base pos after step"
    np.testing.assert_allclose(np.linalg.norm(bkd.get_base_quat(), axis=-1), 1.0, atol=1e-5)
